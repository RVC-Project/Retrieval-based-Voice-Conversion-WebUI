# チケット一覧

> 日本語歌声変換プロジェクトのマイルストーン別チケット
> マイルストーン計画: [milestones.md](../milestones.md)

## 進捗サマリー

| チケット | フェーズ | 工数 | ステータス | GPU |
|---------|---------|------|-----------|-----|
| [M0: 評価基盤構築](M0_evaluation_infrastructure.md) | Week 1 | 3人日 | 実装完了（ベースライン測定除く） | 4070 Ti Super |
| [M1: 即効性改善](M1_immediate_improvements.md) | Week 1-2 | 10人日 | 実装完了（評価実行除く） | 4070 Ti Super |
| [M2-A: SSLモデル統合](M2A_ssl_model_integration.md) | Week 3 | 10.5人日 | 未着手 | 4070 Ti Super |
| [M2-B: 日本語歌声事前学習](M2B_pretrain_japanese_singing.md) | Week 4-5 | 8人日+GPU | 未着手 | 4090 / Cloud |
| [M3-A: 損失関数改善](M3A_loss_function_improvements.md) | Week 6 | 4人日 | 未着手 | 4070 Ti Super |
| [M3-B: ボコーダ改善](M3B_vocoder_improvements.md) | Week 7-8 | 8-9人日+GPU | 未着手 | 4090 / Cloud |
| [M4: 高度な最適化](M4_advanced_optimization.md) | Week 9+ | 数ヶ月 | 未着手 | A100推奨 |

## 依存関係

```
Week 1-2: M0 + M1（並列実行可能）
              ↓ Go/No-Go判定①
Week 3:   M2-A（SSLモデル統合）← M0必須
              ↓
Week 4-5: M2-B（事前学習）← M2-A完了必須
              ↓ Go/No-Go判定②
Week 6:   M3-A（損失関数改善）
              ↓
Week 7-8: M3-B（ボコーダ改善）← M3-A完了必須
              ↓ Go/No-Go判定③
Week 9+:  M4（オプション）
```

## ハードウェア構成

| 用途 | GPU | VRAM |
|------|-----|------|
| 開発・実装・推論テスト | RTX 4070 Ti Super | 16GB |
| ファインチューニング・学習 | RTX 4090 | 24GB |
| 事前学習（M2/M3） | Cloud GPU | 必要に応じ |

## チケットの構成

各チケットには以下のセクションが含まれます:

1. **タスク目的とゴール** — 何を、なぜ作るのか
2. **実装する内容の詳細** — サブタスク、変更対象ファイル、技術仕様
3. **エージェントチーム構成** — 役割と人数
4. **提供範囲・テスト項目** — スコープ、ユニットテスト、E2Eテスト
5. **懸念事項とレビュー項目** — リスクとレビューチェックリスト
6. **一から作り直すとしたら** — 理想設計・思想
7. **後続タスクへの連絡事項** — 引き継ぎ情報

## レビュー履歴

全チケットはエージェントチーム（作成エージェント+レビューエージェント）で作成・検証済み。

### チケット作成時のレビュー修正:

- **M1**: `configs/config.py` の `preprocess_per` 変更漏れを追加（致命的）
- **M2-A**: 工数を8→10.5人日に修正、milestones.mdも同期更新
- **M2-B**: `train.py` の `strict=False` 矛盾を解消、GPU名をRTX 4090に統一
- **M3-A**: DWT周波数帯域の重み付けロジック修正、EMA保存パス3箇所の完全化
- **M3-B**: CQTDiscriminatorインターフェース修正、mel_processing.pyキャッシュバグ発見
- **M4**: F0量子化ビン256のハードコード箇所を推論4ファイルで追加発見

### 実装後のレビュー修正（M0/M1）:

- **M0**: DTW radius 1→20、MCD n_mels 128→40、Whisper large-v3デフォルト化、日本語正規化拡充、テスト71件に拡充
- **M1**: Discriminator weight_decay=0、c_mrstft 2.5→5.0、MRSTFT win_lengths修正、演歌f0_max=900、アニソンf0_max=1200、y_hat_mel.to(amp_dtype)バグ修正
