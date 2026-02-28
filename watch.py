import torch
import pygame
from game import GameEnv
from ai import ConvSNN, HIDDEN_SIZE
import sys
import os

FRAME_SKIP = 2  # 與 train.py 保持一致


def watch_ai(episode_number):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 支援 'final' 或數字
    if str(episode_number) == "final":
        model_path = "snn_ep_final.pth"
    else:
        model_path = f"snn_ep_{episode_number}.pth"

    if not os.path.exists(model_path):
        print(f"找不到 {model_path}！請等訓練跑到那個存檔點。")
        return

    print(f"載入模型：{model_path}")
    model = ConvSNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    env    = GameEnv(render_mode=True)
    states = env.reset()

    # GRU hidden states for both players
    h_p1 = torch.zeros(1, 1, HIDDEN_SIZE, device=device)
    h_p2 = torch.zeros(1, 1, HIDDEN_SIZE, device=device)

    done = False
    step = 0
    print("開始觀戰！(關閉視窗可退出)")

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        with torch.no_grad():
            s1 = torch.tensor(states[0], dtype=torch.float32).unsqueeze(0).to(device)
            s2 = torch.tensor(states[1], dtype=torch.float32).unsqueeze(0).to(device)

            logits1, _, h_p1 = model(s1, h_p1)
            logits2, _, h_p2 = model(s2, h_p2)

            probs1 = torch.sigmoid(logits1[0])
            probs2 = torch.sigmoid(logits2[0])

            action1 = (probs1 > 0.5).float().cpu().tolist()
            action2 = (probs2 > 0.5).float().cpu().tolist()

        states, rewards, done, _ = env.step(action1, action2, frame_skip=FRAME_SKIP)
        step += 1

        env.render(info=f"Episode:{episode_number} | Step:{step} | "
                        f"R1:{rewards[0]:.1f} R2:{rewards[1]:.1f}")

    pygame.quit()
    print("本局觀戰結束！")


if __name__ == "__main__":
    target_ep = "final"
    if len(sys.argv) > 1:
        try:
            target_ep = int(sys.argv[1])
        except ValueError:
            target_ep = sys.argv[1]

    watch_ai(target_ep)
