import torch
from app.models import Wav2Lip

class Wav2LipModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Wav2Lip()
        checkpoint = torch.load('app/checkpoints/wav2lip_gan.pth', map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def inference(self, face, mel):
        with torch.no_grad():
            face = torch.FloatTensor(face).to(self.device)
            mel = torch.FloatTensor(mel).to(self.device)
            face = face.unsqueeze(0)
            mel = mel.unsqueeze(0)
            output = self.model(mel, face)
            return output.cpu().numpy()