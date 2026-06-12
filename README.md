# Cantor Virtual — Síntese de Voz Cantada Personalizada (SVS)

Projeto da disciplina **Processamento de Áudio e Voz** (UFG).

**O que faz:** qualquer pessoa grava alguns minutos da própria voz e o sistema sintetiza
**essa voz cantando** uma música fornecida como **partitura (melodia + letra)** — e,
opcionalmente, anima um **avatar (rosto)** cantando o resultado.

```
 partitura (MIDI/MusicXML + letra)
        │
        ▼
 [1] front-end de partitura  ──►  fonemas + notas + durações
        │
        ▼
 [2] DiffSinger + NSF-HiFiGAN ──►  voz-guia cantada (timbre base)
        │
        ▼
 [3] RVC v2 (treinado na sua voz) ──►  a MESMA performance no SEU timbre
        │
        ▼
 [4] mix opcional com instrumental (Demucs) ──►  .wav final
        │
        ▼
 [5] avatar opcional: SadTalker (foto + áudio) ──►  .mp4 (rosto cantando)
```

## Escopo ético e legal (leia antes de usar)

Este projeto **só** sintetiza vozes de pessoas que consentem — na prática, a voz de quem
grava as próprias amostras. Veja [`CONSENT.md`](CONSENT.md).

- ❌ **Não** clona a voz de artistas/pessoas reais sem consentimento (isso é deepfake de voz).
- ❌ **Não** anima a imagem de pessoa real sem consentimento (isso é deepfake visual).
- ❌ **Não** baixa nem treina em gravações comerciais protegidas por direitos autorais.
- ✅ A música entra como **partitura** (melodia + letra), não como gravação master.
- ✅ O repertório de demonstração usa músicas de **domínio público / licença aberta**.
- ✅ O avatar usa um rosto **sintético/fictício**, próprio, ou de voluntário que consente.

Para demonstrar com uma música específica, transcreva a melodia e a letra para um arquivo
MIDI/MusicXML (como em qualquer cover feito em aula) — o sistema é genérico e funciona com
qualquer partitura.

## Requisitos

- **Python 3.10** para o stack de ML (RVC/`fairseq`, DiffSinger não têm wheels estáveis
  no 3.11+). O front-end de partitura (`src/score`) roda em qualquer 3.10–3.13.
- GPU NVIDIA com drivers CUDA (ver VRAM recomendada em cada etapa)
- `ffmpeg` instalado e no PATH
- `git` no PATH (para clonar os repositórios DiffSinger/RVC)

> ⚠️ **Atenção:** a máquina de desenvolvimento atual tem Python 3.13. Crie um ambiente
> **3.10** dedicado antes de instalar o stack de ML. No Windows, instale o Python 3.10 e:
> ```powershell
> py -3.10 -m venv .venv
> ```

## Instalação

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1

# PyTorch com CUDA (escolha o build da sua GPU; exemplo CUDA 12.4):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
python scripts/setup_models.py       # baixa pesos pré-treinados + clona repos (one-shot)
python scripts/make_demo_song.py     # gera a música de demonstração (domínio público)
```

### O que já está validado vs. o que precisa do ambiente 3.10 + GPU

| Etapa | Status | Precisa de |
|---|---|---|
| Front-end de partitura (`src/score`) | ✅ testado | só `pretty_midi`, `music21`, `g2p_en` |
| Importador/validador (`scripts/import_score.py`) | ✅ testado | idem |
| Transcrição automática AMT (`src/transcribe`, `scripts/transcribe.py`) | ✅ testado | `librosa` (roda no 3.13) |
| Conversão p/ formato DiffSinger (`src/svs/ds_format`) | ✅ testado | nada extra |
| Detecção de hardware / mixagem / E/S | ✅ testado | `torch` (detecção), `soundfile` (E/S) |
| Síntese SVS (DiffSinger) | ⏳ requer setup | Python 3.10 + GPU + pesos |
| Treino/conversão de timbre (RVC) | ⏳ requer setup | Python 3.10 + GPU + pesos |
| Separação de fontes (Demucs) | ⏳ requer setup | `demucs` instalado |
| Avatar visual (SadTalker) | ⏳ requer setup | Python 3.10 + GPU + checkpoints |

## Uso

```powershell
# 1. Obter uma partitura (escolha UMA opção):
#    a) importar um MusicXML/MIDI que VOCÊ exportou (ex.: do MuseScore)
python scripts/import_score.py minha_musica.musicxml --name minha_musica
#    b) transcrever automaticamente a partir de VOCÊ cantando/tocando a melodia
python scripts/transcribe.py minha_melodia.wav --name minha_musica --lyrics letra.txt

# 2. Treinar o modelo de voz a partir das suas gravações (>= 5 min em data/voices/<nome>/)
python scripts/train_voice.py --voice data/voices/meu_nome

# 3. Sintetizar a partitura na sua voz
python -m src.pipeline --song data/songs/minha_musica --voice meu_nome --out out/demo.wav

# 3b. (opcional) com avatar: rosto cantando o áudio  ->  gera out/demo.mp4
python scripts/get_face.py --name meu_avatar         # rosto sintético (não-real)
python -m src.pipeline --song data/songs/minha_musica --voice meu_nome \
    --avatar-image data/faces/meu_avatar.jpg --out out/demo.wav

# 4. Ou usar a interface gráfica
python app/gradio_app.py
```

### Obtendo a partitura — só material que você tem direito de usar

A música entra como **partitura (melodia + letra)**, nunca como gravação comercial. Fontes
legítimas: uma transcrição que **você** fez (MuseScore), a sua própria voz/instrumento
transcritos automaticamente (`scripts/transcribe.py`), domínio público ou licença aberta.
**Não** use o transcritor sobre gravações comerciais de terceiros nem baixe partituras de
catálogos protegidos. Veja [`CONSENT.md`](CONSENT.md).

## Estrutura

| Caminho | Papel |
|---|---|
| `src/score/` | parse de MIDI/MusicXML + letra → fonemas alinhados às notas |
| `src/transcribe/` | AMT: áudio (sua voz/instrumento) → MIDI melódico |
| `src/svs/` | DiffSinger + NSF-HiFiGAN → voz-guia cantada |
| `src/voice/` | treino e inferência RVC (timbre do usuário) |
| `src/separate/` | Demucs (instrumental / limpeza de amostras) |
| `src/avatar/` | SadTalker (foto + áudio → vídeo de rosto cantando) |
| `src/pipeline.py` | orquestra a cadeia completa |
| `app/gradio_app.py` | UI de demonstração |
| `scripts/` | download de modelos, transcrição, treino de voz, rosto sintético |
| `data/songs/` | partituras de demonstração (domínio público) |
| `data/voices/` | gravações dos usuários (não versionado) |
| `data/faces/` | rostos do avatar (não versionado) |

## Créditos / reuso

Construído sobre projetos open-source: **DiffSinger**, **NSF-HiFiGAN**, **RVC**
(Retrieval-based Voice Conversion), **Demucs**, **SadTalker** + **GFPGAN**,
**music21**, **phonemizer**, **librosa**.
