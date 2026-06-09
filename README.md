# Cantor Virtual — Síntese de Voz Cantada Personalizada (SVS)

Projeto da disciplina **Processamento de Áudio e Voz** (UFG).

**O que faz:** qualquer pessoa grava alguns minutos da própria voz e o sistema sintetiza
**essa voz cantando** uma música fornecida como **partitura (melodia + letra)**.

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
```

## Escopo ético e legal (leia antes de usar)

Este projeto **só** sintetiza vozes de pessoas que consentem — na prática, a voz de quem
grava as próprias amostras. Veja [`CONSENT.md`](CONSENT.md).

- ❌ **Não** clona a voz de artistas/pessoas reais sem consentimento (isso é deepfake de voz).
- ❌ **Não** baixa nem treina em gravações comerciais protegidas por direitos autorais.
- ✅ A música entra como **partitura** (melodia + letra), não como gravação master.
- ✅ O repertório de demonstração usa músicas de **domínio público / licença aberta**.

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
| Conversão p/ formato DiffSinger (`src/svs/ds_format`) | ✅ testado | nada extra |
| Detecção de hardware / mixagem / E/S | ✅ testado | `torch` (detecção), `soundfile` (E/S) |
| Síntese SVS (DiffSinger) | ⏳ requer setup | Python 3.10 + GPU + pesos |
| Treino/conversão de timbre (RVC) | ⏳ requer setup | Python 3.10 + GPU + pesos |
| Separação de fontes (Demucs) | ⏳ requer setup | `demucs` instalado |

## Uso

```powershell
# 1. Treinar o modelo de voz a partir das suas gravações (>= 5 min em data/voices/<nome>/)
python scripts/train_voice.py --voice data/voices/meu_nome

# 2. Sintetizar uma partitura na sua voz
python -m src.pipeline --song data/songs/demo --voice meu_nome --out out/demo.wav

# 3. Ou usar a interface gráfica
python app/gradio_app.py
```

## Estrutura

| Caminho | Papel |
|---|---|
| `src/score/` | parse de MIDI/MusicXML + letra → fonemas alinhados às notas |
| `src/svs/` | DiffSinger + NSF-HiFiGAN → voz-guia cantada |
| `src/voice/` | treino e inferência RVC (timbre do usuário) |
| `src/separate/` | Demucs (instrumental / limpeza de amostras) |
| `src/pipeline.py` | orquestra a cadeia completa |
| `app/gradio_app.py` | UI de demonstração |
| `scripts/` | download de modelos e treino de voz |
| `data/songs/` | partituras de demonstração (domínio público) |
| `data/voices/` | gravações dos usuários (não versionado) |

## Créditos / reuso

Construído sobre projetos open-source: **DiffSinger**, **NSF-HiFiGAN**, **RVC**
(Retrieval-based Voice Conversion), **Demucs**, **music21**, **phonemizer**.
