
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import gradio as gr
from tensorflow.keras.models import load_model

modelname = "modelo_01_UnetUS"
model = load_model(f'Modelos/{modelname}.keras')

def gen_dict_cores_labels():
    mask_dict = {
        (0, 0, 0): (0, "Fundo"),             # Preto = None
        (100, 0, 100): (1, "Figado"),        # Violeta = Liver (fígado)
        (255, 255, 255): (2, "Osso"),       # Branco = Bone (Osso)
        (0, 255, 0): (3, "Vesicula biliar"),    # Verde = Gallbladder (Vesícula biliar)
        (255, 255, 0): (4, "Rim"),       # Amarelo = Kidney (Rins)
        (0, 0, 255): (5, "Pancreas"),       # Azul = Pancreas
        (255, 0, 0): (6, "Veia"),        # Vermelho = Vessels (Veias)
        (255, 0, 255): (7, "Baço"),       # Rosa = Spleen (Baço)
        (0, 255, 255): (8, "Gland. Adrenal")       # Azul claro = Adrenal (Glândula Adrenal)
    }
    return mask_dict

# Função para converter de one-hot para RGB e identificar classes
def onehot_to_rgb_labels(oneHot, dictCoresLabels):
    oneHot = np.array(oneHot)
    oneHot = np.argmax(oneHot, axis=-1)
    output = np.zeros(oneHot.shape + (3,))  # Cria a matriz RGB de saída
    present_classes = set()

    for color, (_, label) in dictCoresLabels.items():
        mask = oneHot == _ # Ajustado para evitar erro
        output[mask] = color
        if np.any(mask):
            present_classes.add(label)

    return np.uint8(output), list(present_classes)

# Função para criar legenda em HTML com nomes e cores
def create_legend_html(present_labels, dictCoresLabels):
    legend_html = "<ul style='list-style-type:none; padding:0;'>"
    for color, (_, label) in dictCoresLabels.items():
        if label in present_labels:
            color_hex = f"rgb{color}"  # Converte a cor para o formato RGB
            legend_html += f"<li style='color:{color_hex}; font-weight:bold;'>{label}</li>"
    legend_html += "</ul>"
    return legend_html

def predictmask_legend(image):
    dictCoresLabels = gen_dict_cores_labels()

    original_size = image.size
    image_resized = preprocess_image(image)

    y_pred = model.predict(image_resized)
    y_pred = y_pred[0]

    # Converte a predição para RGB e identifica os rótulos
    y_pred_rgb, labels = onehot_to_rgb_labels(y_pred, dictCoresLabels)

    # Redimensiona a máscara para o tamanho original da imagem
    y_pred_resized = cv2.resize(y_pred_rgb, original_size, interpolation=cv2.INTER_NEAREST)
    image_original_rgb = image.convert("RGB")

    # Combina a imagem original com a máscara
    alpha = 0.9
    beta = 0.5
    image_combined = cv2.addWeighted(np.array(image_original_rgb), beta, y_pred_resized, alpha, 0)
    legend_html = create_legend_html(labels, dictCoresLabels) #gera legenda

    return Image.fromarray(np.uint8(image_combined)), legend_html

def preprocess_image(image):

    image_resized = cv2.resize(np.array(image), (256, 256))
    image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

    # Normaliza e add batchdim
    image_resized = image_resized.astype('float32') / 255.0
    image_resized = np.expand_dims(image_resized, axis=0)

    # Adiciona uma dimensão extra p/ canal de cor
    image_resized = np.expand_dims(image_resized, axis=-1)

    return image_resized

# Interface Gradio
interface = gr.Interface(
    fn=predictmask_legend,   # Função para predição e legendas
    inputs=gr.Image(type="pil", label="Carregue uma imagem"),  # Entrada de imagem
    outputs=[
        gr.Image(type="pil", label="Imagem Segmentada"),  # Saída da imagem com sobreposição
        gr.HTML(label="Legenda dos Órgãos Detectados")                # Saída em HTML para legenda colorida
    ],
    title="Segmentador de Ultrassom Abdominal",         # Título da interface
    description="Carregue uma ultrassonografia de cavidade abdominal para identificar os órgãos."
)

interface.launch(share=True)