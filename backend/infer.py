import torch


class RVCInfer:

    def __init__(self, net_g, device):
        self.net_g = net_g
        
        
        self.device = device



    @torch.no_grad()
    def infer(self, phone, pitch, pitchf, sid=0):
        

        phone = torch.from_numpy(phone).float().to(self.device)
        pitch = torch.from_numpy(pitch).float().unsqueeze(0).to(self.device)
        pitchf = torch.from_numpy(pitchf).float().unsqueeze(0).to(self.device)

        phone_lengths = torch.LongTensor([phone.shape[1]]).to(self.device)

        sid = torch.tensor([sid], dtype=torch.long, device=self.device)

        print("phone:", phone.shape)
        print("pitch:", pitch.shape)
        print("pitchf:", pitchf.shape)
        print("phone_lengths:", phone_lengths) 
        print("sid:", sid.shape)

        audio = self.net_g.infer(
                phone,
                phone_lengths,
                pitch,
                pitchf,
                sid
            )
        
        audio = audio.squeeze().cpu().numpy().astype("float32")

        return audio
