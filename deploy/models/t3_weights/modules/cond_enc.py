from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn, Tensor

from .perceiver import Perceiver
from .t3_config import T3Config


@dataclass
class T3Cond(nn.Module):
    speaker_emb: Tensor = torch.ones(0)
    clap_emb: Tensor = torch.ones(0)
    cond_prompt_speech_tokens: Tensor = torch.ones(0)
    cond_prompt_speech_emb: Tensor = torch.ones(0)
    emotion_adv: Tensor = 0.5 * torch.ones(1, 1)

    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            if v is None:
                v = torch.ones(0)
            elif k == 'cond_prompt_speech_emb' and len(v.shape) == 3:
                v = v[0]
            elif k == 'emotion_adv' and len(v.shape) == 3:
                v = v[0]
            setattr(self, k, v)

    def save(self, fpath):
        torch.save(self.__dict__, fpath)

    @staticmethod
    def load(fpath, map_location="cpu"):
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return T3Cond(**kwargs)

    def to(self, device):
        self.speaker_emb = self.speaker_emb.to(device)
        self.clap_emb = self.clap_emb.to(device)
        self.cond_prompt_speech_tokens = self.cond_prompt_speech_tokens.to(device)
        self.cond_prompt_speech_emb = self.cond_prompt_speech_emb.to(device)
        self.emotion_adv = self.emotion_adv.to(device)
        return self


class T3CondEnc(nn.Module):
    def __init__(self, hp: T3Config):
        super().__init__()
        self.hp = hp
        if hp.encoder_type == "voice_encoder":
            self.spkr_enc = nn.Linear(hp.speaker_embed_size, hp.n_channels)
        else:
            raise NotImplementedError(str(hp.encoder_type))

        self.emotion_adv_fc = None
        if hp.emotion_adv:
            self.emotion_adv_fc = nn.Linear(1, hp.n_channels, bias=False)

        self.perceiver = None
        if hp.use_perceiver_resampler:
            self.perceiver = Perceiver()

    def forward(self, cond: T3Cond):
        assert (cond.cond_prompt_speech_tokens.shape == (0,)) == (cond.cond_prompt_speech_emb.shape == (0,))

        cond_spkr = self.spkr_enc(cond.speaker_emb.view(self.hp.speaker_embed_size))
        cond_spkr = cond_spkr.unsqueeze(0)

        empty = torch.zeros(0, cond_spkr.shape[-1], device=cond_spkr.device, dtype=cond_spkr.dtype)
        assert cond.clap_emb.shape == (0,), "clap_embed not implemented"
        cond_clap = empty

        cond_prompt_speech_emb = cond.cond_prompt_speech_emb
        if cond_prompt_speech_emb.shape == (0,):
            cond_prompt_speech_emb = empty
        elif self.hp.use_perceiver_resampler:
            cond_prompt_speech_emb = self.perceiver(cond_prompt_speech_emb)

        cond_emotion_adv = empty
        if self.hp.emotion_adv:
            assert cond.emotion_adv.shape != (0,)
            cond_emotion_adv = self.emotion_adv_fc(cond.emotion_adv)

        cond_embeds = torch.cat((
            cond_spkr,
            cond_clap,
            cond_prompt_speech_emb,
            cond_emotion_adv,
        ), dim=0)
        return cond_embeds
