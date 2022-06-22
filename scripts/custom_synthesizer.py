import os
import random
import librosa
import torch
import math
import time
import sys
import logging

import numpy as np

from textwrap import wrap
import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker
import matplotlib.animation as animation
import soundfile as sf
#from mpl_toolkits import mplot3d
import subprocess

from model import embedding_net, multimodal_context_net, seq2seq_net, speech2gesture

from utils.data_utils import remove_tags_marks, convert_dir_vec_to_pose, dir_vec_pairs

sys.path.insert(0, '../../../gentle')
import gentle

gentle_resources = gentle.Resources()

# TODO: avoid saving and loading audio files so often
# currently: TTS saves as tts_output.wav and loads as binary for sending (sending data instead of filename between services? / sending audio only could be removed since sending video with sound?)
# here this file is loaded using librosa
# contents are resampled and saved again in align_words to be used by gentle (is resampling necessary?/ maybe gentle can be replaced anyway)
# create_video_and_save saves audio file again to merge with video using ffmpeg
# gentle seems to require 8K, ffmpeg and code 16K, TTS doesnt care when saving
# making sure TTS saves at 16K creates dependency but allows to remove re-save in create_video_and_save

# speed up: main problem is rendering video: done saving video, took 9.8 seconds

class GestureSynthesizer():

    def __init__(self,
            checkpoint_path,
            device=None,
            result_save_path = './generated_files'
        ):
        self.device = device
        if device is None:
            self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        self.result_save_path = result_save_path
        os.makedirs(result_save_path, exist_ok=True)
        self.args, self.generator, self.lang_model = self.load_checkpoint_and_model(checkpoint_path)
        self.mean_dir_vec = np.array(self.args.mean_dir_vec).squeeze()


        
    def generate_and_save(self, text, audio_filename='../tts_output.wav'):
        result_filename = 'generated_gestures'
        audio, audio_sr = librosa.load(audio_filename, mono=True, sr=16000, res_type='kaiser_fast')
        print('audio_sr should be 16000', audio_sr)
        text_without_tags = remove_tags_marks(text)
        words_with_timestamps = self.align_words(audio, text_without_tags)
        dir_vec = self.generate_gestures(self.args, self.generator, self.lang_model, audio, words_with_timestamps)
        out_pos = self.create_video_and_save(
            self.result_save_path, result_filename, dir_vec, self.mean_dir_vec, 'video title', audio=audio,
            clipping_to_shortest_stream=True, delete_audio_file=False)

        return self.result_save_path + '/' + result_filename + '.mp4'


    def load_checkpoint_and_model(self, checkpoint_path):
        print('loading checkpoint {}'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        args = checkpoint['args']
        epoch = checkpoint['epoch']
        lang_model = checkpoint['lang_model']
        speaker_model = checkpoint['speaker_model']
        pose_dim = checkpoint['pose_dim']
        print('epoch {}'.format(epoch))

        generator = multimodal_context_net.PoseGenerator(args,
                        n_words=lang_model.n_words,
                        word_embed_size=args.wordembed_dim,
                        word_embeddings=lang_model.word_embedding_weights,
                        z_obj=speaker_model,
                        pose_dim=pose_dim).to(self.device)
        generator.load_state_dict(checkpoint['gen_dict'])

        # set to eval mode
        generator.train(False)

        return args, generator, lang_model

    def align_words(self, audio, text):
        # downsample audio from 16K to 8K
        audio_8k = librosa.resample(audio, 16000, 8000)#, res_type='kaiser_fast'
        wave_file = 'output/temp.wav'
        sf.write(wave_file, audio_8k, 8000, 'PCM_16')
        # TODO: delete temp.wav again when done? maybe don't takes time

        # run gentle to align words
        aligner = gentle.ForcedAligner(gentle_resources, text, nthreads=2, disfluency=False,
                                    conservative=False)
        gentle_out = aligner.transcribe(wave_file, logging=logging)
        words_with_timestamps = []
        for i, gentle_word in enumerate(gentle_out.words):
            if gentle_word.case == 'success':
                words_with_timestamps.append([gentle_word.word, gentle_word.start, gentle_word.end])
            elif 0 < i < len(gentle_out.words) - 1:
                words_with_timestamps.append([gentle_word.word, gentle_out.words[i-1].end, gentle_out.words[i+1].start])

        return words_with_timestamps

    def generate_gestures(self, args, pose_decoder, lang_model, audio, words, audio_sr=16000, vid=None, seed_seq=None):
        # TODO: inspect vid and args.z_type
        # TODO: use self.args?
        # TODO: use previous as seed_seq
        out_list = []
        n_frames = args.n_poses
        clip_length = len(audio) / audio_sr

        # pre seq
        pre_seq = torch.zeros((1, n_frames, len(args.mean_dir_vec) + 1))
        if seed_seq is not None:
            pre_seq[0, 0:args.n_pre_poses, :-1] = torch.Tensor(seed_seq[0:args.n_pre_poses])
            pre_seq[0, 0:args.n_pre_poses, -1] = 1  # indicating bit for seed poses

        # divide into synthesize units and do synthesize
        unit_time = args.n_poses / args.motion_resampling_framerate
        stride_time = (args.n_poses - args.n_pre_poses) / args.motion_resampling_framerate
        if clip_length < unit_time:
            num_subdivision = 1
        else:
            num_subdivision = math.ceil((clip_length - unit_time) / stride_time) + 1
        audio_sample_length = int(unit_time * audio_sr)

        # prepare speaker input
        if args.z_type == 'speaker':
            if not vid:
                vid = random.randrange(pose_decoder.z_obj.n_words)
            print('vid:', vid, 'from', pose_decoder.z_obj.n_words) # e.g.993 from 1370
            vid = torch.LongTensor([vid]).to(self.device)
        else:
            vid = None

        print('{}, {}, {}, {}, {}'.format(num_subdivision, unit_time, clip_length, stride_time, audio_sample_length))

        out_dir_vec = None
        start = time.time()
        for i in range(0, num_subdivision):
            start_time = i * stride_time
            end_time = start_time + unit_time

            # prepare audio input
            audio_start = math.floor(start_time / clip_length * len(audio))
            audio_end = audio_start + audio_sample_length
            in_audio = audio[audio_start:audio_end]
            if len(in_audio) < audio_sample_length:
                in_audio = np.pad(in_audio, (0, audio_sample_length - len(in_audio)), 'constant')
            in_audio = torch.from_numpy(in_audio).unsqueeze(0).to(self.device).float()

            # prepare text input
            word_seq = self.get_words_in_time_range(word_list=words, start_time=start_time, end_time=end_time)
            extended_word_indices = np.zeros(n_frames)  # zero is the index of padding token
            word_indices = np.zeros(len(word_seq) + 2)
            word_indices[0] = lang_model.SOS_token
            word_indices[-1] = lang_model.EOS_token
            frame_duration = (end_time - start_time) / n_frames
            for w_i, word in enumerate(word_seq):
                print(word[0], end=', ')
                idx = max(0, int(np.floor((word[1] - start_time) / frame_duration)))
                extended_word_indices[idx] = lang_model.get_word_index(word[0])
                word_indices[w_i + 1] = lang_model.get_word_index(word[0])
            print(' ')
            in_text_padded = torch.LongTensor(extended_word_indices).unsqueeze(0).to(self.device)
            #in_text = torch.LongTensor(word_indices).unsqueeze(0).to(self.device)

            # prepare pre seq
            if i > 0:
                pre_seq[0, 0:args.n_pre_poses, :-1] = out_dir_vec.squeeze(0)[-args.n_pre_poses:]
                pre_seq[0, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
            pre_seq = pre_seq.float().to(self.device)

            # synthesize
            print(in_text_padded)

            out_dir_vec, *_ = pose_decoder(pre_seq, in_text_padded, in_audio, vid)

            out_seq = out_dir_vec[0, :, :].data.cpu().numpy()

            # smoothing motion transition
            if len(out_list) > 0:
                last_poses = out_list[-1][-args.n_pre_poses:]
                out_list[-1] = out_list[-1][:-args.n_pre_poses]  # delete last 4 frames

                for j in range(len(last_poses)):
                    n = len(last_poses)
                    prev = last_poses[j]
                    next = out_seq[j]
                    out_seq[j] = prev * (n - j) / (n + 1) + next * (j + 1) / (n + 1)

            out_list.append(out_seq)
        time_taken = time.time() - start
        print('generation took {:.2} s on avg per subdivision and {:.2} s in total'.format(time_taken / num_subdivision, time_taken))

        # aggregate results
        out_dir_vec = np.vstack(out_list)

        return out_dir_vec

    def get_words_in_time_range(self, word_list, start_time, end_time):
        words = []
        for word in word_list:
            _, word_s, word_e = word[0], word[1], word[2]
            if word_s >= end_time:
                break
            if word_e <= start_time:
                continue
            words.append(word)
        return words

    def create_video_and_save(self, save_path, save_filename, output, mean_data, title,
                          audio=None, clipping_to_shortest_stream=False, delete_audio_file=True):
        print('rendering a video...')
        start = time.time()

        fig = plt.figure(figsize=(8, 4))
        axes = [fig.add_subplot(1, 2, 1, projection='3d')]
        axes[0].view_init(elev=20, azim=-60)

        fig.suptitle('\n'.join(wrap(title, 75)), fontsize='medium')

        # un-normalization and convert to poses
        mean_data = mean_data.flatten()
        output = output + mean_data
        output_poses = convert_dir_vec_to_pose(output)

        def animate(i):
            pose = output_poses[i]
            
            if pose is not None: # TODO: should never be None probably remove this check
                axes[0].clear()
                for j, pair in enumerate(dir_vec_pairs):
                    axes[0].plot([pose[pair[0], 0], pose[pair[1], 0]],
                                [pose[pair[0], 2], pose[pair[1], 2]],
                                [pose[pair[0], 1], pose[pair[1], 1]],
                                zdir='z', linewidth=5)
                axes[0].set_xlim3d(-0.5, 0.5)
                axes[0].set_ylim3d(0.5, -0.5)
                axes[0].set_zlim3d(0.5, -0.5)
                axes[0].set_xlabel('x')
                axes[0].set_ylabel('z')
                axes[0].set_zlabel('y')
                axes[0].set_title('{} ({}/{})'.format('generated', i + 1, len(output)))

        num_frames = len(output)
        ani = animation.FuncAnimation(fig, animate, interval=30, frames=num_frames, repeat=False)

        # re-save audio
        audio_path = None
        if audio is not None:
            assert len(audio.shape) == 1  # 1-channel, raw signal
            audio = audio.astype(np.float32)
            sr = 16000
            audio_path = '{}/{}.wav'.format(save_path, save_filename)
            sf.write(audio_path, audio, sr)

        # save video
        try:
            video_path = '{}/temp_{}.mp4'.format(save_path, save_filename)
            ani.save(video_path, fps=15, dpi=80)  # dpi 150 for a higher resolution
            del ani
            plt.close(fig)
        except RuntimeError:
            assert False, 'RuntimeError'

        # merge audio and video
        if audio is not None:
            merged_video_path = '{}/{}.mp4'.format(save_path, save_filename)
            cmd = ['ffmpeg', '-loglevel', 'panic', '-y', '-i', video_path, '-i', audio_path, '-strict', '-2',
                merged_video_path]
            if clipping_to_shortest_stream:
                cmd.insert(len(cmd) - 1, '-shortest')
            subprocess.call(cmd)
            if delete_audio_file:
                os.remove(audio_path)
            os.remove(video_path)

        print('done saving video, took {:.1f} seconds'.format(time.time() - start))
        return output_poses