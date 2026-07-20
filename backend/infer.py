import numpy as np
import torch


class RVCInfer:

    def __init__(self, net_g, device):
        self.net_g = net_g
        
        
        self.device = device



    @torch.no_grad()
    def infer(self, phone, pitch, pitchf, sid=0):
        
        length = min(
            phone.shape[1],
            pitch.shape[0],
            pitchf.shape[0]
        )
        
        phone = phone[:, :length]
        pitch = pitch[:length]
        pitchf = pitchf[:length]


        phone = torch.from_numpy(phone).float().to(self.device)
        pitch = torch.from_numpy(pitch).long().unsqueeze(0).to(self.device)
        pitchf = torch.from_numpy(pitchf).float().unsqueeze(0).to(self.device)

        phone_lengths = torch.LongTensor([phone.shape[1]]).to(self.device)

        sid = torch.tensor([sid], dtype=torch.long, device=self.device)

        #print("phone:", phone.shape)
        #print("pitch:", pitch.shape)
        #print("pitchf:", pitchf.shape)
        #print("phone_lengths:", phone_lengths) 
        #print("sid:", sid.shape)

        audio = self.net_g.infer(
                phone,
                phone_lengths,
                pitch,
                pitchf,
                sid
            )
        
        
        
        #print("INFER AUDIO STATE: ", type(audio))
        #print("INFER AUDIO STATE: ", audio.shape)
        #print("INFER AUDIO STATE: ", audio.min(), audio.max())
        
        audio = audio.squeeze().cpu().numpy().astype("float32")

        return audio
