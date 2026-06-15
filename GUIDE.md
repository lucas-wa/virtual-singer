# Passo a passo — rodar o Cantor Virtual

Duas opções. **Recomendo o Colab** (GPU grátis, sem dor de cabeça com CUDA/Python no Windows).

---

## Opção A — Google Colab (recomendada)

### 1. Abrir o notebook com GPU
1. Suba este projeto para o Google Drive **ou** para um repositório no GitHub (veja o passo 2).
2. Abra `notebooks/virtual_singer_colab.ipynb` no Colab
   (https://colab.research.google.com → *Upload* ou *GitHub*).
3. Menu **Runtime → Change runtime type → Hardware accelerator: GPU (T4)** → *Save*.

### 2. Levar o projeto para o Colab (escolha UM jeito)

**(a) Via GitHub (mais limpo)** — no seu PC, dentro da pasta do projeto:
```powershell
git add -A
git commit -m "cantor virtual: scaffold + score/AMT"
# crie um repo vazio no GitHub e:
git remote add origin https://github.com/SEU_USUARIO/virtual-singer.git
git push -u origin main
```
No notebook, a primeira célula clona esse repositório.

**(b) Via upload .zip** — compacte a pasta do projeto e, no notebook, use a célula de
*upload* (ela descompacta automaticamente). Sem precisar de GitHub.

**(c) Via Google Drive** — copie a pasta para o seu Drive e monte o Drive no notebook.

### 3. Rodar as células na ordem
O notebook está dividido em blocos:
1. **Checar GPU** — confirma a T4.
2. **Obter o projeto** — clona/descompacta/monta (conforme 2a/2b/2c).
3. **Instalar deps leves + demo** — roda partitura, AMT e gera a música de teste.
   ✅ *Isto já funciona e prova o pipeline ponta a ponta (sem voz ainda).*
4. **Instalar PyTorch + clonar DiffSinger/RVC + baixar pesos** — prepara os motores de ÁUDIO.
5. **Treinar a sua voz (RVC)** — você sobe ~5 min da sua voz.
6. **Sintetizar** — gera a sua voz cantando a partitura.
7. **Avatar** — ⛔ **NÃO roda no Colab grátis** (ver abaixo). Esta seção é só referência
   para você rodar localmente / em host permitido.

> ⛔ **Restrição do Colab grátis:** o Google **proíbe** ferramentas de animação facial/
> deepfake (SadTalker) e **encerra a sessão** se você as clonar/rodar. Por isso o avatar
> ficou FORA das células de áudio e o `setup_models.py` **não** baixa o SadTalker por padrão
> (use `--with-avatar` só fora do Colab grátis). O áudio (DiffSinger/RVC) é permitido.
>
> As células 4–6 são as que ajustamos juntos: dependendo das versões do RVC/DiffSinger
> que o Colab baixar, os comandos em `src/svs/engine.py` e `src/voice/rvc.py` podem
> precisar de pequenos ajustes de flags. Rode e me mande o erro que aparecer.

### 4. Usar
- **Partitura:** suba um `.musicxml` (do MuseScore) ou transcreva sua voz com a célula de AMT.
- **Sua voz:** suba 5–15 min de áudio limpo (voz solo).
- **Resultado:** a célula de síntese gera um `.wav` para baixar/ouvir no próprio notebook.

---

## Opção B — Local (Windows, Python 3.10)

Use se você tem uma GPU NVIDIA no PC e prefere rodar local.

### 1. Instalar Python 3.10 (seu sistema tem 3.13; o stack de ML precisa do 3.10)
```powershell
winget install Python.Python.3.10
```

### 2. Criar e ativar o ambiente
```powershell
cd D:\projects\ufg\pav\virtual-singer
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Instalar PyTorch com CUDA + dependências
```powershell
# escolha o build CUDA da sua GPU (exemplo CUDA 12.4):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 4. Confirmar que a GPU foi detectada
```powershell
python -m src.hardware     # deve mostrar o nome da GPU e a VRAM, não "CPU"
```

### 5. Baixar modelos + gerar a demo
```powershell
python scripts/setup_models.py
python scripts/make_demo_song.py
```

### 6. Fluxo completo
```powershell
# (opcional) transcrever sua voz numa partitura:
python scripts/transcribe.py minha_melodia.wav --name minha_musica --lyrics letra.txt

# escolher a voz (alterne quando quiser trocando o --voice):
#   (a) voz pronta da VocalSet (testar sem gravar):
python scripts/get_sample_voice.py --singer female1
python scripts/train_voice.py --voice data/voices/vocalset_female1
#   (b) ou a sua voz (>= 5 min em data/voices/meu_nome/):
python scripts/train_voice.py --voice data/voices/meu_nome

# sintetizar (só áudio) — escolha a voz no --voice:
python -m src.pipeline --song data/songs/minha_musica --voice meu_nome --out out/demo.wav

# (opcional) com avatar visual -> gera out/demo.mp4:
python scripts/get_face.py --name meu_avatar      # rosto sintético (não-real)
python -m src.pipeline --song data/songs/minha_musica --voice meu_nome \
    --avatar-image data/faces/meu_avatar.jpg --out out/demo.wav

# ou a interface gráfica:
python app/gradio_app.py
```

---

## Opção C — Cluster SLURM (H100 + Singularity) — recomendada para o pipeline completo

Tem acesso a uma H100 via SLURM? É o melhor ambiente: **sem a proibição de deepfake do
Colab** (o avatar roda aqui), Python sob seu controle e GPU sobrando.

1. **Imagem do container** (uma vez) — PyTorch + CUDA 12 + **Python 3.10** (fairseq exige 3.10):
   ```bash
   singularity pull /raid/user_lucasalves/images/pytorch_2.2_cu121_py310.sif \
       docker://pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime
   ```
2. **Coloque o repositório** em `$HOST_RAID/$PROJECT_SUBDIR` (ex.: `/raid/user_lucasalves/virtual-singer`).
3. (Opcional) `cp slurm/.env.example slurm/.env` e preencha o `HF_TOKEN`.
4. **Edite os parâmetros** no bloco do `slurm/run_virtual_singer.sbatch` (SONG, VOICE_NAME,
   VOICE_SOURCE, ENGINE, AVATAR_IMAGE) e submeta:
   ```bash
   sbatch slurm/run_virtual_singer.sbatch
   ```

O job: instala deps (+fairseq), baixa modelos (conforme ENGINE/AVATAR), e roda
`scripts/run_all.sh` ponta a ponta: voz pronta (VocalSet) → treino RVC → síntese →
[avatar]. Saída em `out/`. Logs em `logs/vsinger_<jobid>.out`.

> Para o avatar no cluster, defina `AVATAR_IMAGE=synthetic` (ou um caminho) no `.sbatch` —
> o `setup_models.py` é chamado com `--with-avatar` automaticamente. Para a síntese de maior
> qualidade, `ENGINE=diffsinger` (chama `--with-diffsinger`; ainda precisa de voicebank EN).

## O que já está garantido vs. o que ainda iteramos

| Funciona hoje (Colab ou local) | Precisa de ajuste fino |
|---|---|
| Partitura → fonemas (`src/score`) | Treino/conversão RVC na GPU (`src/voice/rvc.py`) |
| **Voz-guia DSP embutida (`src/svs/dsp_guide.py`)** | Inferência DiffSinger opcional (`src/svs/engine.py`) |
| Importador/validador + AMT (sua voz → MIDI) | Avatar SadTalker — só fora do Colab grátis |
| Voz de teste VocalSet, demo, detecção de GPU, mixagem | — |

Lembre-se do escopo (ver `CONSENT.md`): só vozes/rostos próprios ou consentidos (ou rosto
sintético), e a música entra como partitura que você tem direito de usar — nunca gravação
comercial, crawler, nem deepfake (de voz ou de imagem) de pessoa real.
