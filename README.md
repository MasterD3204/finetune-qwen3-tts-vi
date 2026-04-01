Bây giờ viết hướng dẫn hoàn chỉnh:                              
                                                                                          
  ---
  3C. Hướng dẫn chạy step-by-step                                                                                                                                                      
   
  ---                                                                                                                                                                                  
  📋 HƯỚNG DẪN HOÀN CHỈNH: Fine-tune Qwen3-TTS cho tiếng Việt

  Tổng quan pipeline

  1. Chuẩn bị data tiếng Việt  ──►  2. Tokenize audio  ──►  3. Stage 1 (LoRA)
                                                                      │
  6. Đánh giá voice cloning  ◄──  5. Export model  ◄──  4. Stage 2 (optional)

  ---
  Bước 1: Yêu cầu dữ liệu tiếng Việt

  1.1 Cấu trúc JSONL thô

  Mỗi dòng trong train_vi_raw.jsonl:
  {"audio": "./data/vi/utt_001.wav", "text": "Xin chào, hôm nay trời đẹp quá.", "ref_audio": "./data/vi/ref_speaker.wav"}
  {"audio": "./data/vi/utt_002.wav", "text": "Tôi muốn đặt một cái bánh mì thịt.", "ref_audio": "./data/vi/ref_speaker.wav"}

  ▎ Lưu ý: ref_audio phải là cùng 1 file cho toàn bộ dataset — đây là file audio tham chiếu giọng của người nói đích.

  1.2 Yêu cầu về dữ liệu

  ┌─────────────────────────────────────────────────┬──────────────┬───────────────────────────────────┐
  │                    Tiêu chí                     │  Tối thiểu   │             Tốt nhất              │
  ├─────────────────────────────────────────────────┼──────────────┼───────────────────────────────────┤
  │ Số utterances (speaker target)                  │ 300–500      │ 1,000–2,000                       │
  ├─────────────────────────────────────────────────┼──────────────┼───────────────────────────────────┤
  │ Số utterances (Vietnamese general, cho Stage 1) │ 2,000        │ 5,000–10,000                      │
  ├─────────────────────────────────────────────────┼──────────────┼───────────────────────────────────┤
  │ Độ dài mỗi utterance                            │ 1–20 giây    │ 3–10 giây                         │
  ├─────────────────────────────────────────────────┼──────────────┼───────────────────────────────────┤
  │ Chất lượng audio                                │ 16kHz, mono  │ 24kHz, mono, studio-quality       │
  ├─────────────────────────────────────────────────┼──────────────┼───────────────────────────────────┤
  │ SNR                                             │ > 15 dB      │ > 25 dB                           │
  ├─────────────────────────────────────────────────┼──────────────┼───────────────────────────────────┤
  │ Đa dạng nội dung                                │ Nhiều chủ đề │ Bao phủ phoneme tiếng Việt đầy đủ │
  ├─────────────────────────────────────────────────┼──────────────┼───────────────────────────────────┤
  │ 6 thanh điệu                                    │ Phân bố đều  │ Mỗi thanh ≥ 15%                   │
  └─────────────────────────────────────────────────┴──────────────┴───────────────────────────────────┘

  1.3 Nguồn dữ liệu Vietnamese TTS phù hợp

  - VIVOS (25h, miễn phí): https://ailab.hcmus.edu.vn/vivos
  - VietTTS (public domains): GitHub search
  - VLSP TTS datasets (nếu có access)
  - Thu âm tự tạo: dùng Audacity, target 24kHz mono WAV

  ---
  Bước 2: Cài đặt dependencies

  # Từ thư mục dự án
  cd /home/misa/Downloads/Qwen3-TTS

  # Cài đặt base requirements
  pip install -e .

  # Cài thêm cho Vietnamese pipeline
  pip install librosa soundfile editdistance

  # Optional: cho eval
  pip install openai-whisper speechbrain

  ---
  Bước 3: Tokenize audio → audio_codes

  3.1 Dataset thuần Vietnamese (speaker fine-tune, Stage 2)

  cd finetuning

  python prepare_data_vi.py \
    --device cuda:0 \
    --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
    --vi_jsonl ./data/vi_speaker_raw.jsonl \
    --output_jsonl ./data/train_vi_speaker_with_codes.jsonl \
    --output_val_jsonl ./data/val_vi_speaker_with_codes.jsonl \
    --val_ratio 0.05

  3.2 Mixed dataset (Vietnamese general + original, Stage 1)

  python prepare_data_vi.py \
    --device cuda:0 \
    --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
    --vi_jsonl ./data/vi_general_raw.jsonl \
    --original_jsonl ./data/original_lang_raw.jsonl \
    --vi_ratio 0.8 \
    --output_jsonl ./data/train_mixed_with_codes.jsonl \
    --output_val_jsonl ./data/val_mixed_with_codes.jsonl

  ▎ Output sẽ hiện thống kê số lượng và reject entries (nếu có audio lỗi).

  ---
  Bước 4: Stage 1 — Language Adaptation (LoRA)

  Mục tiêu: Dạy talker mapping text tiếng Việt → codec tokens, không làm mất giọng gốc.

  cd finetuning

  python sft_vietnamese.py \
    --stage 1 \
    --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --train_jsonl ./data/train_mixed_with_codes.jsonl \
    --output_model_path ./output/vi_stage1 \
    --speaker_name vi_speaker \
    --speaker_slot 3000 \
    --lora_rank 32 \
    --lora_alpha 64.0 \
    --lr 1e-4 \
    --num_epochs 5 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --save_every_n_steps 500

  Hyperparameters Stage 1 — giải thích

  ┌────────────┬─────────┬──────────────────────────────────────────────────┐
  │   Param    │ Giá trị │                      Lý do                       │
  ├────────────┼─────────┼──────────────────────────────────────────────────┤
  │ lora_rank  │ 32      │ Đủ capacity để học language mapping              │
  ├────────────┼─────────┼──────────────────────────────────────────────────┤
  │ lora_alpha │ 64      │ = 2×rank, tỉ lệ learning rate cho LoRA           │
  ├────────────┼─────────┼──────────────────────────────────────────────────┤
  │ lr         │ 1e-4    │ Cao hơn normal vì chỉ train LoRA (~2% params)    │
  ├────────────┼─────────┼──────────────────────────────────────────────────┤
  │ num_epochs │ 5       │ Đủ để converge mà không overfit                  │
  ├────────────┼─────────┼──────────────────────────────────────────────────┤
  │ batch_size │ 2       │ Với gradient_accumulation=4: effective batch = 8 │
  ├────────────┼─────────┼──────────────────────────────────────────────────┤
  │ vi_ratio   │ 0.8     │ Tránh forgetting ngôn ngữ gốc                    │
  └────────────┴─────────┴──────────────────────────────────────────────────┘

  Theo dõi training

  # Xem tensorboard logs
  tensorboard --logdir ./output/vi_stage1

  # Loss target: primary loss < 2.5 sau 3 epochs
  # Nếu loss không giảm sau 2 epochs: giảm lr xuống 5e-5

  ---
  Bước 5: Stage 2 — Voice Integration (tùy chọn)

  Mục tiêu: Tích hợp giọng người nói đích với tiếng Việt.
  Chỉ cần nếu bạn muốn một predefined speaker tiếng Việt thay vì voice cloning.

  python sft_vietnamese.py \
    --stage 2 \
    --init_model_path ./output/vi_stage1/checkpoint-epoch-4 \
    --train_jsonl ./data/train_vi_speaker_with_codes.jsonl \
    --output_model_path ./output/vi_stage2 \
    --speaker_name vi_speaker \
    --speaker_slot 3000 \
    --lr 5e-6 \
    --num_epochs 3 \
    --batch_size 2 \
    --gradient_accumulation_steps 4

  Hyperparameters Stage 2

  ┌────────────┬─────────┬───────────────────────────────────────────────────────┐
  │   Param    │ Giá trị │                         Lý do                         │
  ├────────────┼─────────┼───────────────────────────────────────────────────────┤
  │ lr         │ 5e-6    │ Rất thấp để tránh catastrophic forgetting             │
  ├────────────┼─────────┼───────────────────────────────────────────────────────┤
  │ num_epochs │ 3       │ Ít epochs hơn, chỉ để refine                          │
  ├────────────┼─────────┼───────────────────────────────────────────────────────┤
  │ stage      │ 2       │ Full model train (không LoRA), speaker encoder frozen │
  └────────────┴─────────┴───────────────────────────────────────────────────────┘

  ---
  Bước 6: Kiểm tra voice cloning

  6.1 Quick test (informal)

  # test_vi_voice_clone.py
  import torch
  from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
  import soundfile as sf

  model = Qwen3TTSModel.from_pretrained(
      "./output/vi_stage1/checkpoint-epoch-4",  # hoặc stage2
      torch_dtype=torch.bfloat16,
  )

  # Test 1: Voice cloning tiếng Việt
  prompt = model.create_voice_clone_prompt("./data/vi/ref_speaker.wav")

  vi_texts = [
      "Xin chào, tôi là trợ lý giọng nói tiếng Việt.",
      "Hôm nay thời tiết thật đẹp, bầu trời trong xanh.",
      "Mười hai trăm nghìn đồng một ki lô gam.",
  ]

  for i, text in enumerate(vi_texts):
      audio, sr = model.generate_voice_clone(
          text=text,
          voice_clone_prompt=prompt,
          language="vi",
      )
      sf.write(f"./test_vi_{i:02d}.wav", audio, sr)
      print(f"Generated: test_vi_{i:02d}.wav")

  # Test 2: Kiểm tra ngôn ngữ gốc vẫn hoạt động
  orig_texts = [
      "Hello, this is a test of the voice cloning system.",  # English
  ]
  for i, text in enumerate(orig_texts):
      audio, sr = model.generate_voice_clone(
          text=text,
          voice_clone_prompt=prompt,
          language="Auto",
      )
      sf.write(f"./test_orig_{i:02d}.wav", audio, sr)

  6.2 Formal evaluation (speaker similarity score)

  # Chuẩn bị file test texts
  cat > eval/vi_test_texts.txt << 'EOF'
  Xin chào, tôi là một hệ thống chuyển văn bản thành giọng nói.
  Hôm nay trời đẹp, nắng vàng trải dài trên những cánh đồng xanh.
  Tôi muốn đặt một vé máy bay đi Hà Nội vào ngày mười lăm tháng ba.
  Giá vàng hôm nay tăng lên một triệu hai trăm nghìn đồng một chỉ.
  EOF

  python eval_voice_cloning.py \
    --model_path ./output/vi_stage1/checkpoint-epoch-4 \
    --ref_audio ./data/vi/ref_speaker.wav \
    --test_texts_vi ./eval/vi_test_texts.txt \
    --output_dir ./eval_results \
    --speaker_name vi_speaker \
    --use_voice_clone \
    --compute_cer

  6.3 Ngưỡng đánh giá

  ┌────────────────────┬───────────────────┬─────────────────────┐
  │ Speaker Similarity │     Đánh giá      │      Hành động      │
  ├────────────────────┼───────────────────┼─────────────────────┤
  │ ≥ 0.80             │ ✅ Xuất sắc       │ Tiếp tục deploy     │
  ├────────────────────┼───────────────────┼─────────────────────┤
  │ 0.70–0.80          │ ✅ Tốt            │ Có thể dùng được    │
  ├────────────────────┼───────────────────┼─────────────────────┤
  │ 0.65–0.70          │ ⚠️  Chấp nhận được │ Cân nhắc thêm data  │
  ├────────────────────┼───────────────────┼─────────────────────┤
  │ < 0.65             │ ❌ Degraded       │ Điều chỉnh strategy │
  └────────────────────┴───────────────────┴─────────────────────┘

  ---
  Các tình huống xử lý sự cố

  ❌ Voice cloning bị mất (similarity < 0.65)

  Nguyên nhân có thể:
  1. Quá ít Vietnamese data → model overfit, quên base weights
  2. LR quá cao ở Stage 2

  Giải pháp:
  - Tăng vi_ratio lên 0.9 trong mixed data
  - Giảm lr Stage 2 xuống 1e-6
  - Tăng số epochs Stage 1 LoRA (thêm 2–3 epochs)

  ❌ Phát âm tiếng Việt sai (CER > 0.3)

  Nguyên nhân có thể:
  1. Không đủ Vietnamese data
  2. Data chất lượng kém (nhiễu, accent lạ)

  Giải pháp:
  - Kiểm tra rejected.jsonl xem có entries bị filter không
  - Thêm ít nhất 2000 utterances tiếng Việt chất lượng cao
  - Verify text normalization: normalize_vietnamese_text("123.456đ") → "một trăm hai mươi ba nghìn bốn trăm năm mươi sáu đồng"

  ❌ Training loss không giảm

  Nguyên nhân:
  - Batch size quá nhỏ, gradient noise cao

  Giải pháp:
  - Tăng gradient_accumulation_steps lên 8
  - Giảm lr xuống 5e-5

  ❌ CUDA OOM

  Giải pháp:
  - Giảm batch_size xuống 1
  - Tăng gradient_accumulation_steps lên 8
  - Dùng --attn_implementation eager nếu flash_attention_2 gây vấn đề

  ---
  Tóm tắt các file được tạo

  finetuning/
  ├── sft_12hz.py              # Script gốc (giữ nguyên)
  ├── sft_vietnamese.py        # ← MỚI: script fine-tune tiếng Việt với LoRA
  ├── prepare_data.py          # Script gốc (giữ nguyên)
  ├── prepare_data_vi.py       # ← MỚI: chuẩn bị data + text normalization
  ├── dataset.py               # Giữ nguyên (không cần sửa)
  └── eval_voice_cloning.py    # ← MỚI: đánh giá speaker similarity & CER

  ---
  Tổng kết kiến trúc quyết định

  ┌─────────────────────────────────────────────────────────────────┐
  │                   WHY THIS STRATEGY WORKS                        │
  ├─────────────────────────────────────────────────────────────────┤
  │                                                                   │
  │  Voice Cloning Preservation:                                     │
  │  • Speaker Encoder: FROZEN → giọng extraction không bị ảnh hưởng│
  │  • Speaker embedding slot (pos 6): vẫn giữ nguyên mechanism      │
  │  • LoRA → base weights không thay đổi → voice conditioning intact│
  │                                                                   │
  │  Vietnamese Language Learning:                                   │
  │  • LoRA adapters in Q/K/V/O → học text→codec mapping mới        │
  │  • text_embedding trainable → phát âm Việt được học             │
  │  • Language ID "vi" được thêm vào config                        │
  │                                                                   │
  │  Anti-Forgetting:                                                │
  │  • Mixed data (80% vi + 20% original) ở Stage 1                 │
  │  • LoRA không ghi đè base weights → base language vẫn trong đó  │
  │  • Stage 2 với LR = 5e-6 → fine adjustment không phá vỡ         │
  │                                                                   │
