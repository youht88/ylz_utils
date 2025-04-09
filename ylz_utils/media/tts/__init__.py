from __future__ import annotations
from typing import TYPE_CHECKING, Union
if TYPE_CHECKING:
    from ylz_utils.langchain import LangchainLib

import dashscope
from dashscope.audio.tts import SpeechSynthesizer

import dashscope
import sys
from dashscope.api_entities.dashscope_response import SpeechSynthesisResponse
from dashscope.audio.tts import ResultCallback, SpeechSynthesizer, SpeechSynthesisResult

class TTSLib():
    model = None
    def __init__(self,langchainLib:LangchainLib):
        self.langchainLib = langchainLib
        self.config = langchainLib.config
    def init(self):
        api_keys = self.config.get('TTS.DASHSCOPE.API_KEYS')        
        self.model = self.config.get('TTS.DASHSCOPE.MODEL') or 'sambert-zhichu-v1'
        if api_keys:
            api_key = self.langchainLib.split_keys(api_keys)[0]
            dashscope.api_key=api_key
        else:
            raise Exception("请先设置TTS.DASHSCOPE.API_KEYS")

    def tts_save(self,text:str,filename:str):
        self.init()
        import pyaudio
        result = SpeechSynthesizer.call(model=self.model,
                                        text=text,
                                        sample_rate=48000)
        if result.get_audio_data() is not None:
            with open(filename, 'wb') as f:
                f.write(result.get_audio_data())
    def tts_play(self,text:str):
        self.init()
        callback = TTS_Callback()
        import pyaudio
        SpeechSynthesizer.call(model='sambert-zhichu-v1',
                       text=text,
                       sample_rate=48000,
                       format='pcm',
                       callback=callback)

class TTS_Callback(ResultCallback):
    _player = None
    _stream = None

    def on_open(self):
        print('Speech synthesizer is opened.')
        self._player = pyaudio.PyAudio()
        self._stream = self._player.open(
            format=pyaudio.paInt16,
            channels=1, 
            rate=48000,
            output=True)

    def on_complete(self):
        print('Speech synthesizer is completed.')

    def on_error(self, response: SpeechSynthesisResponse):
        print('Speech synthesizer failed, response is %s' % (str(response)))

    def on_close(self):
        print('Speech synthesizer is closed.')
        self._stream.stop_stream()
        self._stream.close()
        self._player.terminate()

    def on_event(self, result: SpeechSynthesisResult):
        if result.get_audio_frame() is not None:
            print('audio result length:', sys.getsizeof(result.get_audio_frame()))
            self._stream.write(result.get_audio_frame())

        if result.get_timestamp() is not None:
            print('timestamp result:', str(result.get_timestamp()))       