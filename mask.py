import os
import cv2
import numpy as np

# ------ CONFIGURAÇÕES ------
IMAGENS_DIR = "imagens"
ANOTACOES_DIR = "anotacoes"
MASKS_DIR = "masks"

if not os.path.exists(MASKS_DIR):
    os.makedirs(MASKS_DIR)

# ------ FUNÇÃO PARA CRIAR A MÁSCARA ------
def criar_mask(nome_imagem):
    nome_base = os.path.splitext(nome_imagem)[0]
    img_path = os.path.join(IMAGENS_DIR, nome_imagem)
    label_path = os.path.join(ANOTACOES_DIR, f"{nome_base}.lines.txt")

    # Carregar imagem para saber tamanho
    img = cv2.imread(img_path)
    height, width = img.shape[:2]

    # Criar máscara preta
    mask = np.zeros((height, width), dtype=np.uint8)

    if not os.path.exists(label_path):
        print(f"Aviso: não existe anotação para {nome_imagem}")
        return

    with open(label_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.strip():
                pontos = list(map(int, line.strip().split()))
                for i in range(0, len(pontos) - 2, 2):
                    pt1 = (pontos[i], pontos[i+1])
                    pt2 = (pontos[i+2], pontos[i+3])
                    cv2.line(mask, pt1, pt2, 255, 4)  # Linha branca com espessura 4

    # Guardar a máscara
    mask_path = os.path.join(MASKS_DIR, f"{nome_base}_mask.png")
    cv2.imwrite(mask_path, mask)
    print(f"Máscara criada: {mask_path}")

# ------ PROCESSAR TODAS AS IMAGENS ------
imagens = [f for f in os.listdir(IMAGENS_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
imagens.sort()

for nome_imagem in imagens:
    criar_mask(nome_imagem)

print("Processo concluído. Máscaras guardadas na pasta 'masks'.")

