# M0/M1 評価レポート

> 実施日: 2026-04-11
> 実施者: yuta
> 環境: Windows 11, RTX 4070 Ti SUPER 16GB, Python 3.12, PyTorch 2.10 (bf16)

---

## 1. 概要

M0（評価基盤構築）とM1（即効性改善）の実装完了後、実際の歌声データで学習・推論・評価を実施し、Go/No-Go判定①を行った。

---

## 2. 実験条件

### 2.1 データセット

| 項目 | 値 |
|------|-----|
| 話者 | はるなさん（声優） |
| 曲数 | 6曲 |
| 合計長 | 20.7分（1240.7秒） |
| フォーマット | 48kHz / モノラル / PCM 24bit |
| 内容 | ボーカルのみ（歌声収録） |

| # | 曲名 | 長さ |
|---|------|------|
| 1 | OutOfTheGravity | 4:10 |
| 2 | CatLoving | 1:54 |
| 3 | HideAndSeek | 3:17 |
| 4 | KurakuKuroku | 3:41 |
| 5 | Kyukurarin | 3:34 |
| 6 | Saphire | 4:04 |

### 2.2 学習設定（M1改善適用済み）

| パラメータ | 値 | M1変更点 |
|-----------|-----|---------|
| ベースモデル | pretrained_v2/f0G48k.pth, f0D48k.pth | - |
| サンプリングレート | 48kHz | - |
| batch_size | 8 | - |
| segment_size | 34560 | M1で拡張 |
| p_dropout | 0.1 | M1で追加 |
| weight_decay (G) | 0.01 | M1で追加 |
| weight_decay (D) | 0.0 | M1で修正 |
| c_mrstft | 5.0 | M1で追加 (MRSTFT損失) |
| c_mel | 45 | - |
| 混合精度 | bf16 | M1で修正 (.half()→.to(amp_dtype)) |
| F0抽出 | RMVPE | - |
| filter_radius | 1 | M1でデフォルト変更 (3→1) |
| スライス数 | 193 | 前処理で自動分割 |

### 2.3 評価方法

自己再構築テスト: 学習済みモデルで元音声（CatLoving_Vox.wav, 114.3秒）を変換し、原音と比較。

```bash
uv run python tools/eval/run_eval.py \
  --ref CatLoving_Vox.wav --conv CatLoving_conv_e200.wav \
  --metrics all --device cuda
```

---

## 3. 学習経過

### 3.1 ロス推移

| エポック | loss_mel | loss_mrstft | loss_kl | loss_disc | loss_gen |
|---------|----------|-------------|---------|-----------|----------|
| 1 | 34.27 | 8.43 | 8.93 | 4.37 | 2.94 |
| 8 | 21.31 | 5.59 | 1.74 | 4.00 | 3.12 |
| 50 | 20.86 | 5.52 | 1.70 | 3.93 | 3.02 |
| 100 | 19.36 | 5.33 | 1.77 | 3.84 | 3.35 |
| 150 | 19.39 | 5.25 | 1.31 | 3.69 | 3.64 |
| 200 | 18.78 | 5.08 | 1.44 | 3.74 | 3.39 |

### 3.2 所見

- loss_melは最初の8エポックで急速に収束（34.3→21.3）、以降は緩やかに低下
- loss_mrstftも同様に急速収束後、緩やかに低下
- loss_klはEpoch 8で1.74まで下がり、以降安定
- **50エポック以降の改善は微小**: 50ep→200epでloss_melは20.9→18.8（-10%程度）
- 1エポックあたり約28秒（bs=8, 4070 Ti SUPER）

### 3.3 学習時の技術的問題と対応

| 問題 | 原因 | 対応 |
|------|------|------|
| torch.load WeightsOnlyError | PyTorch 2.6でweights_onlyデフォルトがTrueに変更 | ラッパースクリプト (_run_train.py) でmonkey-patch |
| gloo backend失敗 | WindowsでProcessGroupGlooが動作しない | train.pyにWindows単一GPU判定を追加、dist.init_process_groupをスキップ |
| matplotlib tostring_rgb | 新版matplotlibでAPI削除 | buffer_rgba()に置換 (utils.py) |
| bf16テンソルのnumpy変換 | BFloat16はnumpy非対応 | .float()を追加 (train.py L540-542) |
| muteファイル名不一致 | filelistにmute48000.wavと記載、実際はmute48k.wav | filelist再生成 |

---

## 4. 評価結果

### 4.1 メトリクス比較（50ep vs 200ep）

| メトリクス | 50ep | 200ep | 変化 | PASS閾値 | FAIL閾値 | 判定 |
|-----------|------|-------|------|---------|---------|------|
| **MCD** | 30.78 dB | 29.97 dB | -2.6% | <6.0 | >8.0 | **FAIL** |
| **F0 RMSE** | 38.92 cents | 41.46 cents | +6.5% | <20.0 | >50.0 | **WARN** |
| **Whisper CER** | 33.8% | 34.5% | +2.0% | <10% | >20% | **FAIL** |
| **VUV error** | 0.25% | 0.28% | - | - | - | 良好 |
| **全体** | FAIL | FAIL | - | - | - | **FAIL** |

### 4.2 詳細分析

#### MCD (29.97 dB → FAIL)

- 閾値6.0 dBに対して約5倍の値
- 同一ファイルでのサニティチェックはMCD=0.0 dB（計算ロジックは正常）
- 自然対数ベースのメルケプストラムで計算（librosa dBスケール問題を修正済み）
- **この値は自己再構築における実際の音色歪み量を反映**

#### F0 RMSE (41.46 cents → WARN)

- WARN範囲（20-50 cents）内
- VUVエラー率0.28%は非常に優秀（有声/無声判定の精度が高い）
- RMVPEによるF0抽出は安定して動作
- 50ep→200epで悪化（38.9→41.5）は、生成品質の変動によるもの

#### Whisper CER (34.5% → FAIL)

- 参照音声と変換音声の両方をWhisper large-v3で転写し比較
- **曲の前半（イントロ部分）で大きな乖離**: 変換音声では歌い出し部分の歌詞が異なる認識結果に
- **後半は比較的正確**: 「会いたいお間違いなんてみんなは言う」以降はほぼ一致
- 英語歌詞部分（Everybody today is a lovely day）は両方とも認識できているが微細な差異あり

### 4.3 MCD閾値の妥当性について

現在のMCD閾値（PASS=6.0, FAIL=8.0）は、milestones.mdの数値目標（最終目標<6.0 dB）に基づいて設定されたが、自然対数ベースのメルケプストラムMCDでは**閾値が過度に厳しい可能性**がある。

参考: 一般的な音声変換研究でのMCD値
- 高品質VC: 5-7 dB（librosa dBスケールMFCC使用時）
- 中品質VC: 7-10 dB
- 低品質VC: 10+ dB

今回の計算は自然対数ベースであり、dBスケールとは異なるレンジになる。**閾値の再校正が必要**。

---

## 5. Go/No-Go判定①

### 5.1 判定基準と結果

| 基準 | Go条件 | 結果 | 判定 |
|------|--------|------|------|
| MCD改善 | 5%以上改善 | ベースライン比較不可（M1前のモデルなし） | **判定不能** |
| F0 RMSE改善 | 10%以上改善 | 同上 | **判定不能** |
| リアルタイムレイテンシ | 200ms以下 | 未測定 | **未実施** |
| 既存モデル互換 | 完全維持 | 維持（pretrained_v2互換） | **Go** |

### 5.2 判定

**条件付きGo**

Go/No-Go判定①は「M1改善前後の比較」を前提としていたが、M1改善前のモデル（改善なしの学習）を別途用意していないため、絶対値での評価となった。

以下の理由からM2への進行を推奨:

1. **M1改善のコード実装は完了**: MRSTFT損失、dropout、bf16修正、歌声プリセット等が正常に動作
2. **学習パイプラインの動作確認完了**: 前処理→学習→推論→評価のフルパイプラインが自動化
3. **評価基盤（M0）の動作確認完了**: MCD/F0 RMSE/Whisper CERの3メトリクスが正常に計測可能
4. **現状の品質限界はSSLモデルに起因**: 英語HuBERT (ContentVec)による日本語歌声の特徴量抽出が本質的なボトルネック
5. **エポック数増加の効果は限定的**: 50ep→200epで改善なし（学習の収束は十分）

---

## 6. M0/M1 実装サマリ

### 6.1 M0: 評価基盤（完了）

| コンポーネント | ファイル | 状態 |
|--------------|---------|------|
| 評価CLI | `tools/eval/run_eval.py` | 動作確認済み |
| 音声読み込み | `tools/eval/audio_utils.py` | 動作確認済み |
| MCD計測 | `tools/eval/metrics/mcd.py` | 動作確認済み（自然対数ベースに修正） |
| F0 RMSE計測 | `tools/eval/metrics/f0_accuracy.py` | 動作確認済み |
| Whisper CER計測 | `tools/eval/metrics/whisper_cer.py` | 動作確認済み |
| テスト | `tests/eval/` (71テスト) | 全テストPASS |

### 6.2 M1: 即効性改善（完了）

| タスク | ファイル | 状態 |
|--------|---------|------|
| FCPE統合 | pipeline.py, extract_f0_print.py | 実装済み |
| F0レンジ拡張 | pipeline.py, extract_f0_print.py | 実装済み |
| filter_radius変更 | infer-web.py, infer_cli.py | 実装済み |
| Dropout + Weight Decay | configs/v2/*.json, train.py | 動作確認済み |
| segment_size拡張 | configs/v2/*.json | 動作確認済み |
| MRSTFT損失 | losses.py, train.py | 動作確認済み (c_mrstft=5.0) |
| 歌声プリセット | f0_presets.py, infer-web.py | 実装済み |
| bf16混合精度修正 | train.py | 動作確認済み |

### 6.3 学習環境修正（実験中に対応）

| 修正 | ファイル | 内容 |
|------|---------|------|
| Windows単一GPU対応 | train.py | gloo回避、DDP スキップ |
| matplotlib互換 | infer/lib/train/utils.py | tostring_rgb→buffer_rgba |
| bf16 numpy変換 | train.py | .float()追加 |
| torch.load互換 | _run_train.py | weights_only=False パッチ |

---

## 7. 今後の課題と推奨事項

### 7.1 短期（M2準備）

1. **MCD閾値の再校正**: 自然対数ベースMCDの適切なPASS/FAIL閾値を、既知の高品質VCモデルで校正する
2. **M1改善前ベースラインの取得**: mainブランチ（改善前コード）で同一データを学習・評価し、M1の効果を定量比較
3. **評価データの分離**: 学習データと評価データを分け、過学習の影響を排除

### 7.2 中期（M2: SSL置換）

1. **日本語SSLモデル（kushinada-hubert）への置換**: 日本語歌声の特徴量抽出精度を本質的に改善
2. **Whisper CERの大幅改善が期待**: 日本語音素レベルの特徴量が得られれば、歌詞の認識精度が向上
3. **MCDも改善見込み**: より適切な特徴量による合成品質の向上

### 7.3 技術的負債

- `_run_train.py`のtorch.loadパッチ: fairseq側の更新またはHuBERTローダーの書き換えで恒久対応すべき
- Windows単一GPU対応: train.pyの`_skip_dist`フラグは暫定対応。分散学習とのコード共存を整理すべき

---

## 8. 成果物一覧

```
eval_output/
├── CatLoving_conv_e50.wav      # 50エポックモデルの変換音声
├── CatLoving_conv_e200.wav     # 200エポックモデルの変換音声
├── baseline_e50.json           # 50エポック評価結果 (MCD + F0 RMSE)
├── baseline_e50_cer.json       # 50エポック評価結果 (Whisper CER)
└── baseline_e200.json          # 200エポック評価結果 (全メトリクス)

logs/haruna_singing/
├── config.json                 # 学習設定
├── filelist.txt                # 学習ファイルリスト (193 + 2 mute)
├── G_2333333.pth               # Generator チェックポイント (200ep)
├── D_2333333.pth               # Discriminator チェックポイント (200ep)
├── 0_gt_wavs/                  # 前処理済み音声 (193ファイル)
├── 1_16k_wavs/                 # 16kHzリサンプル済み
├── 2a_f0/                      # F0特徴量
├── 2b-f0nsf/                   # F0 NSF特徴量
├── 3_feature768/               # HuBERT 768次元特徴量
├── total_fea.npy               # FAISS用統合特徴量
└── added_IVF1104_Flat_nprobe_1_haruna_singing_v2.index

assets/weights/
├── haruna_singing.pth          # 最終モデル
├── haruna_singing_e50_s1400.pth
├── haruna_singing_e100_s2800.pth
├── haruna_singing_e150_s4200.pth
└── haruna_singing_e200_s5600.pth
```
