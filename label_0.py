import pygame
import os
import sys

# ------ CONFIGURAÇÕES ------
IMAGENS_DIR = "imagens"
ANOTACOES_DIR = "anotacoes"

SCREEN_SIZE = (1280, 720)
RAIO_PONTO = 15

# ------ SETUP ------
pygame.init()
screen = pygame.display.set_mode(SCREEN_SIZE)
pygame.display.set_caption("Labeling Tool - Etapa 3 completa")

font = pygame.font.SysFont("Arial", 24)

# ------ CORES ------
CORES_LINHAS = [
    (0, 255, 0),    # Linha 1 - Verde
    (0, 0, 255),    # Linha 2 - Azul
    (255, 0, 0)     # Linha 3 - Vermelho
]

# ------ FUNÇÕES ------
def carregar_imagens(pasta):
    suportados = [".jpg", ".jpeg", ".png", ".bmp"]
    imagens = [f for f in os.listdir(pasta) if os.path.splitext(f)[1].lower() in suportados]
    imagens.sort()
    return imagens

def mostrar_imagem(nome_imagem):
    path = os.path.join(IMAGENS_DIR, nome_imagem)
    img = pygame.image.load(path)
    img = pygame.transform.scale(img, SCREEN_SIZE)
    screen.blit(img, (0, 0))

    texto = font.render(nome_imagem, True, (255, 255, 255))
    screen.blit(texto, (10, 10))

def desenhar_linhas(pontos):
    for idx, linha in enumerate(pontos):
        cor = CORES_LINHAS[idx]
        if len(linha) > 1:
            pygame.draw.lines(screen, cor, False, linha, 3)
        for pt in linha:
            pygame.draw.circle(screen, (255, 255, 255), pt, RAIO_PONTO)

def guardar_anotacoes(nome_imagem, pontos):
    nome_base = os.path.splitext(nome_imagem)[0]
    path = os.path.join(ANOTACOES_DIR, f"{nome_base}.lines.txt")


    with open(path, "w") as f:
        for linha in pontos:
            if linha:
                line_str = " ".join(f"{x} {y}" for (x, y) in linha)
                f.write(line_str + "\n")
            else:
                f.write("\n")  # Linha vazia se não houver pontos

    print(f"Anotação guardada em: {path}")


def carregar_anotacoes(nome_imagem):
    nome_base = os.path.splitext(nome_imagem)[0]
    path = os.path.join(ANOTACOES_DIR, f"{nome_base}.lines.txt")
    pontos = [[], [], []]

    if os.path.exists(path):
        with open(path, "r") as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                if idx < 3:
                    values = list(map(float, line.strip().split()))
                    pontos[idx] = [(int(values[i]), int(values[i+1])) for i in range(0, len(values), 2)]
    return pontos

# ------ PREPARAR ------
if not os.path.exists(ANOTACOES_DIR):
    os.makedirs(ANOTACOES_DIR)

imagens = carregar_imagens(IMAGENS_DIR)
if not imagens:
    print("Não foram encontradas imagens na pasta:", IMAGENS_DIR)
    sys.exit()

indice = 0
linha_atual = 0
pontos = carregar_anotacoes(imagens[indice])

dragging = False
drag_index = None
drag_linha = None

# ------ LOOP PRINCIPAL ------
running = True
while running:
    screen.fill((0, 0, 0))
    mostrar_imagem(imagens[indice])
    desenhar_linhas(pontos)

    texto_linha = font.render(f"Linha atual: {linha_atual + 1}", True, (255, 255, 0))
    screen.blit(texto_linha, (10, 40))

    pygame.display.flip()

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            print(event.button)
            if event.button == 1:  # Esquerdo → adicionar ponto na linha atual
                linha = pontos[linha_atual]
                if len(linha) < 100:
                    linha.append(pos)

            elif event.button == 3:  # Direito → procurar ponto em todas as linhas
                dragging = False
                drag_index = None
                drag_linha = None

                for idx_linha, linha in enumerate(pontos):
                    for idx_pt, pt in enumerate(linha):
 
                        if (pos[0] - pt[0]) ** 2 + (pos[1] - pt[1]) ** 2 <= RAIO_PONTO ** 2:
 
                            dragging = True
                            drag_index = idx_pt
                            drag_linha = idx_linha
                            break


        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 3:
                dragging = False
                drag_index = None
                drag_linha = None

        elif event.type == pygame.MOUSEMOTION:
            if dragging and drag_index is not None and drag_linha is not None:
                pos = pygame.mouse.get_pos()
                pontos[drag_linha][drag_index] = pos

        elif event.type == pygame.KEYDOWN:

            if event.key == pygame.K_RIGHT:
                indice = (indice + 1) % len(imagens)
                pontos = carregar_anotacoes(imagens[indice])
                linha_atual = 0

            elif event.key == pygame.K_LEFT:
                indice = (indice - 1) % len(imagens)
                pontos = carregar_anotacoes(imagens[indice])
                linha_atual = 0

            elif event.key == pygame.K_SPACE:
                linha_atual = (linha_atual + 1) % 3

            elif event.key == pygame.K_c:
                pontos[linha_atual] = []

            elif event.key == pygame.K_BACKSPACE:
                if pontos[linha_atual]:
                    pontos[linha_atual].pop()

            elif event.key == pygame.K_RETURN:
                guardar_anotacoes(imagens[indice], pontos)
                indice = (indice + 1) % len(imagens)
                pontos = carregar_anotacoes(imagens[indice])
                linha_atual = 0

            elif event.key == pygame.K_q:
                running = False

pygame.quit()

