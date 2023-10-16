from train import model
from train import decode
import torch


if __name__ == '__main__':
    model.load_state_dict(torch.load('model_weight_with_single_head_final.pth'))
    model.eval()

    # # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long)
    x = model.generate(context, max_new_tokens=200)[0].tolist()
    print(decode(x))
