# 日本語歌声データセット調査レポート

> RVC学習用途での利用可能性を中心に調査（2026-04-10）
>
> **ステータス**: 本調査レポートは日本語歌声データセットの調査結果であり、コード実装を伴わない。M0で評価基盤（MCD-DTW, F0 RMSE, Whisper CER）が構築済みのため、データセットの品質評価は自動化可能。M1で前処理パラメータ（threshold=-38dB, min_length=2000ms, per=5.0s, overlap=0.5）が歌声向けに最適化済みであり、本レポートのデータセットを学習に利用する準備が整っている。日本語歌声事前学習の実施はM2-B以降。

## 前提: RVCに歌詞テキストは不要

RVCの学習パイプラインは音声WAVファイルのみを入力とする:

```
音声WAV → HuBERT特徴量(.npy自動抽出) + F0ピッチ(.npy自動抽出) + スペクトログラム
```

歌詞テキスト、音素ラベル、MIDI、MusicXMLなどのアノテーションは一切不要。
データセット選定基準は**音声品質**と**ライセンス**のみ。

---

## 1. RVC学習に直接使用可能な日本語歌声データセット

### Tier 1: 最優先（アカペラ・高品質・ライセンス良好）

#### JVS-MuSiC（日本語多話者歌声コーパス）

| 項目 | 詳細 |
|------|------|
| URL | https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_music |
| 歌唱者 | **100名**（男性49名、女性51名） |
| 曲数 | 各歌唱者2曲（共通曲「かたつむり」+個別曲、計約200録音） |
| サンプリングレート | 24kHz |
| 形式 | アカペラ（ボーカルのみ）、UVR5分離不要 |
| ジャンル | 童謡 |
| ライセンス | **個人利用・商用利用可**。再配布不可。一部アップロードは3ファイルまで可 |
| 品質 | スタジオ録音 |
| RVC適性 | **★★★★★** マルチスピーカー事前学習に最適 |

**推奨用途**: 多話者事前学習。100名分の声質バリエーションにより汎化性能向上。

---

#### ~~PJS（音素バランス日本語歌声コーパス）~~ — 除外

> **除外理由**: CC BY-SA 4.0ライセンスのShareAlike条項により、モデル全体にSA条項が波及するリスクがあるため使用しない。

---

#### NIT-SONG070（名工大歌唱DB）

| 項目 | 詳細 |
|------|------|
| URL | https://sourceforge.net/projects/sinsy/files/ |
| 歌唱者 | 1名（女性） |
| 曲数 | 約70曲 |
| 総時間 | 約1.2時間 |
| サンプリングレート | 48kHz |
| 形式 | アカペラ、UVR5分離不要 |
| ジャンル | 童謡・唱歌 |
| ライセンス | **CC BY 3.0（商用利用可、帰属表示必須）** |
| 品質 | スタジオ録音 |
| RVC適性 | **★★★★★** ライセンス最自由 + 十分なデータ量 |

**推奨用途**: 商用利用を見据えた事前学習。ライセンスが最も自由。

---

### Tier 2: 高優先（アカペラ・高品質・研究ライセンス）

#### 東北きりたん歌唱データベース

| 項目 | 詳細 |
|------|------|
| URL | https://zunko.jp/kiridev/login.php |
| ラベルデータ | https://github.com/mmorise/kiritan_singing |
| 歌唱者 | 1名（女性、プロ歌手） |
| 曲数 | 50曲 |
| 総時間 | **約57分** |
| サンプリングレート | **48kHz** / 16bit |
| 形式 | アカペラ、UVR5分離不要 |
| ジャンル | **J-POP**（今風の楽曲） |
| ライセンス | 研究用途のみ。商用不可。クレジット「(c)SSS」必須。再配布不可 |
| 品質 | スタジオ録音 |
| RVC適性 | **★★★★☆** J-POPジャンルが貴重。研究限定が制約 |

**推奨用途**: J-POP歌声の研究用モデル学習。童謡中心のDBと異なり今風の楽曲。

---

#### 東北イタコ歌唱データベース

| 項目 | 詳細 |
|------|------|
| URL | https://zunko.jp/itadev/login.php |
| ラベルデータ | https://github.com/mmorise/itako_singing |
| 歌唱者 | 1名（女性） |
| 曲数 | 約50曲 |
| 総時間 | 約1時間 |
| サンプリングレート | 48kHz |
| 形式 | アカペラ、UVR5分離不要 |
| ジャンル | J-POP |
| ライセンス | 研究用途のみ。商用不可。クレジット「(c)SSS」必須 |
| 品質 | スタジオ録音 |
| RVC適性 | **★★★★☆** きりたんDBの姉妹プロジェクト |

---

#### No.7 歌唱データベース（小岩井ことり）

| 項目 | 詳細 |
|------|------|
| URL | https://voiceseven.com/7dev/login.php |
| ラベルデータ | https://github.com/mmorise/no7_singing |
| 歌唱者 | 1名（声優・小岩井ことり） |
| 曲数 | 約50曲 |
| 総時間 | 約1時間 |
| サンプリングレート | 48kHz |
| 形式 | アカペラ、UVR5分離不要 |
| ジャンル | オリジナル楽曲（小岩井ことり作詞・作曲・歌唱） |
| ライセンス | 研究用途のみ。商用利用は有償で要相談 |
| 品質 | スタジオ録音 |
| RVC適性 | **★★★★☆** プロ声優の歌声 |

---

#### GTSinger 日本語部分（NeurIPS 2024 Spotlight）

| 項目 | 詳細 |
|------|------|
| URL | https://huggingface.co/datasets/GTSinger/GTSinger |
| GitHub | https://github.com/AaronZ345/GTSinger |
| 歌唱者 | 日本語2名（JA-Soprano-1、JA-Tenor-1）/ 全体20名 |
| 総時間 | 全体80.59h（日本語部分は約8時間と推定） |
| サンプリングレート | **48kHz / 24bit** |
| 形式 | アカペラ、UVR5分離不要 |
| ジャンル | 6技法（ミックスボイス、ファルセット、ブレシー、咽頭発声、ビブラート、ポルタメント） |
| ライセンス | **研究目的オープンソース** |
| 品質 | **プロフェッショナルスタジオ録音** |
| RVC適性 | **★★★★☆** 歌唱技法の多様性が貴重。24bitの最高品質 |

**推奨用途**: 歌唱技法（ビブラート・ファルセット等）の学習に特に有効。

---

### Tier 3: 利用可能（追加データとして有用）

#### JSUT-song

| 項目 | 詳細 |
|------|------|
| URL | https://sites.google.com/site/shinnosuketakamichi/publication/jsut-song |
| 歌唱者 | 1名（女性） |
| 曲数 | 27曲 |
| 総時間 | 約25分 |
| サンプリングレート | 48kHz |
| ライセンス | 個人利用可。商用利用は要相談 |
| RVC適性 | **★★★☆☆** データ量が少なめ |

---

#### おふとんP歌声データベース

| 項目 | 詳細 |
|------|------|
| URL | https://sites.google.com/view/oftn-utagoedb |
| 歌唱者 | 1名（**男性**） |
| 曲数 | 49曲 + キー変更3曲 |
| 総時間 | 約61分 |
| 形式 | アカペラ、ピッチ補正・ノイズ除去済み |
| ライセンス | クレジット「DB制作:おふとんP」必須。再配布禁止 |
| RVC適性 | **★★★☆☆** 貴重な男性歌声データ |

---

#### 鬼肉くるみ歌声データベース（ONIKU）

| 項目 | 詳細 |
|------|------|
| URL | https://onikuru.info/db-download/ |
| 歌唱者 | 1名（女性） |
| 曲数 | 56曲 |
| 形式 | アカペラ、ピッチ補正・ノイズ除去済み |
| ライセンス | クレジット「Virtual Singer ONIKU Kurumi」必須 |
| RVC適性 | **★★★☆☆** |

---

#### 波音リツ歌声データベース

| 項目 | 詳細 |
|------|------|
| URL | https://github.com/taroushirani/nnsvs_namine_ritsu_utagoe_db |
| 歌唱者 | 1名 |
| 曲数 | 60曲 |
| ライセンス | 独自利用規約 |
| RVC適性 | **★★★☆☆** |

---

#### OJaMa-Song（一般男性歌声）

| 項目 | 詳細 |
|------|------|
| URL | https://sython.org/Corpus/OJaMa-Song/ |
| 歌唱者 | 1名（20歳男性、非プロ） |
| 曲数 | 27曲（複数スタイル：通常歌唱、ウィスパー、ハミング、読み上げ風） |
| サンプリングレート | **48kHz / 24bit** |
| ライセンス | 研究・分析目的で無料。商用は要相談 |
| RVC適性 | **★★★☆☆** 複数歌唱スタイルがユニーク |

---

#### 夏目悠李歌声データベース

| 項目 | 詳細 |
|------|------|
| URL | https://github.com/AmanoKei/Natsume_Singing |
| 歌唱者 | 1名（男性） |
| 曲数 | 約50曲 |
| ライセンス | 独自利用規約 |
| RVC適性 | **★★★☆☆** 男性歌声 |

---

#### jaCappella Corpus（アカペラ重唱）

| 項目 | 詳細 |
|------|------|
| URL | https://huggingface.co/datasets/jaCappella/jaCappella |
| 歌唱者 | 複数名（6声部構成） |
| 曲数 | 35曲 |
| 総時間 | 約34分 |
| サンプリングレート | 48kHz |
| ジャンル | 童謡を10ジャンル（ジャズ、パンクロック、ボサノバ、演歌等）に編曲 |
| ライセンス | 著作権は制作チーム保持。研究用途での加工音声共有可 |
| 品質 | スタジオ録音、各声部別ファイル |
| RVC適性 | **★★★☆☆** 声部別に分離済みで利用しやすい |

---

## 2. 事前学習・補助用データセット（歌声以外）

| データセット | 規模 | 内容 | URL | RVC用途 |
|-------------|------|------|-----|---------|
| **ReazonSpeech** | 35,000h | 日本語放送音声 | https://huggingface.co/datasets/reazon-research/reazonspeech | HuBERT事前学習（rinna実証済み） |
| **MoeSpeech** | 623h / 473キャラ | アニメ声優の話し声 | https://huggingface.co/datasets/litagin/moe-speech | 声質多様性の事前学習 |
| **JVS** | 30h / 100名 | 通常/ささやき/裏声 | 高道研究室 | マルチスピーカー話し声事前学習 |
| **JSUT** | 10h / 1名 | 読み上げ音声 | 高道研究室 | 補助データ |
| **japanese-voice-combined** | 86,000サンプル | 複数DB統合 | https://huggingface.co/datasets/kadirnar/japanese-voice-combined | 補助データ |

---

## 3. UVR5分離が必要なデータ（伴奏付き）

| データセット | 曲数 | 歌唱者 | ジャンル | URL |
|-------------|------|--------|---------|-----|
| **RWC Popular Music** | 100曲 | 34名 | J-POP | https://staff.aist.go.jp/m.goto/RWC-MDB/ |
| **FruitsMusic** | 40曲 | 多数 | アイドル | https://huggingface.co/datasets/fruits-music/fruits-music |

RWC Popular Musicは研究用途限定・有償だが、J-POPの本格的な楽曲が含まれる貴重なソース。

---

## 4. 既存の日本語RVC事前学習チェックポイント

自前でデータセットから学習する代わりに、既に学習済みのチェックポイントを利用する方法もある。

| チェックポイント | Phone Embedder | 配布先 |
|----------------|---------------|--------|
| `f0X48k768_jphubert_v2` | rinna/jp-hubert-base | [BOOTH](https://booth.pm/ja/items/4802383) |
| `f0X48k768_contentvec_v2` | ContentVec | [BOOTH](https://booth.pm/ja/items/4802383) |
| KLM 4.0 事前学習モデル | 韓日英100h+ | [BOOTH](https://booth.pm/ja/items/5835415) |
| 共通ウェイト | ContentVec | [Kaggle](https://www.kaggle.com/datasets/nadare/rvc-webui-tuned-weights) |

### 最新SSLモデル情報（2025年3月更新）

事前学習の基盤となるSSLモデルについて、2025年3月に産総研から `imprt/kushinada-hubert-base`（62,215h日本語、Apache 2.0）が公開された。rinnaの `japanese-hubert-base`（19,000h）の3.3倍のデータ量で学習されており、RVCのContentVecとドロップイン置換可能（768次元、12層、HuBERT Base）。詳細は [SSLモデル追加調査レポート](research_ssl_models_update.md) を参照。

---

## 5. フリー素材・コミュニティリソース

| リソース | 内容 | URL |
|---------|------|-----|
| つくよみちゃん公式RVCモデル | 学習済みRVCモデル5種。商用利用可 | https://tyc.rei-yumesaki.net/work/software/rvc/ |
| つくよみちゃんUTAU音源 | 歌声合成用音素素材。商用可 | https://tyc.rei-yumesaki.net/material/utau/ |
| あみたろの声素材工房 | フリー音声素材 + RVCモデル | https://amitaro.net/synth/rvc/ |
| BOOTH RVC学習済みデータ | 138点以上のモデル | [BOOTH検索](https://booth.pm/ja/items?tags%5B%5D=RVC+%E5%AD%A6%E7%BF%92%E6%B8%88%E3%81%BF%E3%83%87%E3%83%BC%E3%82%BF) |
| ニコニ・コモンズ | CC素材のボーカル音源 | https://commons.nicovideo.jp/ |
| d-elf.com フリーボーカルBGM | ボーカル入りフリーBGM | https://www.d-elf.com/free-bgm/free-bgm-vo |

---

## 6. 推奨データ戦略

### ユースケース別の推奨構成

#### A. 研究用の日本語歌声変換モデル構築

```
事前学習（マルチスピーカー）:
  JVS-MuSiC (100名) + NIT-SONG070 (1名, CC BY 3.0)
    ↓
歌声適応:
  東北きりたん (50曲J-POP) + GTSinger日本語 (歌唱技法6種)
    ↓
ターゲット話者ファインチューニング:
  声優の歌声データ 10分
```

**合計アカペラデータ量**: 約5-10時間 + ターゲット10分

#### B. 商用利用可能なモデル構築

```
事前学習:
  JVS-MuSiC（商用可）+ NIT-SONG070 (CC BY 3.0)
    ↓
ターゲット話者ファインチューニング:
  声優の歌声データ 10分
```

CC BY / 商用可ライセンスのデータのみで構成。PJS（CC BY-SA）は除外。

#### C. 最大品質を目指すモデル

```
HuBERT事前学習:
  rinna/japanese-hubert-base（ReazonSpeech 19,000h で学習済み）
    ↓
RVCモデル事前学習:
  JVS-MuSiC + NIT-SONG070 + きりたん + イタコ + GTSinger
    ↓
歌声適応:
  全データセット結合（約10-15時間のアカペラ歌声）
    ↓
ターゲット話者:
  声優の歌声データ 10分 + データ拡張で40-100分相当に
```

---

## 7. 全データセット一覧（サマリー）

| # | データセット | 時間 | 話者 | SR | ライセンス | アカペラ | ジャンル | 適性 |
|---|-------------|------|------|-----|-----------|---------|---------|------|
| 1 | **JVS-MuSiC** | - | 100名 | 24k | 商用可 | Yes | 童謡 | ★★★★★ |
| ~~2~~ | ~~PJS~~ | - | - | - | ~~CC BY-SA 4.0~~ | - | - | **除外** |
| 3 | **NIT-SONG070** | 1.2h | 1名F | 48k | CC BY 3.0 | Yes | 童謡 | ★★★★★ |
| 4 | 東北きりたん | 57min | 1名F | 48k | 研究のみ | Yes | J-POP | ★★★★☆ |
| 5 | 東北イタコ | ~1h | 1名F | 48k | 研究のみ | Yes | J-POP | ★★★★☆ |
| 6 | No.7 | ~1h | 1名F | 48k | 研究のみ | Yes | オリジナル | ★★★★☆ |
| 7 | GTSinger(JA) | ~8h | 2名 | 48k/24bit | 研究OS | Yes | 多技法 | ★★★★☆ |
| 8 | JSUT-song | 25min | 1名F | 48k | 個人利用可 | Yes | 童謡 | ★★★☆☆ |
| 9 | おふとんP | 61min | 1名M | 44.1k | 独自規約 | Yes | 童謡/POP | ★★★☆☆ |
| 10 | ONIKU | - | 1名F | - | 独自規約 | Yes | 童謡/POP | ★★★☆☆ |
| 11 | 波音リツ | - | 1名 | - | 独自規約 | Yes | POP/童謡 | ★★★☆☆ |
| 12 | OJaMa-Song | - | 1名M | 48k/24bit | 研究可 | Yes | 童謡 | ★★★☆☆ |
| 13 | 夏目悠李 | ~1h | 1名M | - | 独自規約 | Yes | POP | ★★★☆☆ |
| 14 | jaCappella | 34min | 複数 | 48k | 研究可 | Yes(声部別) | 多ジャンル | ★★★☆☆ |

---

## 参考リンク

- [高道研究室 コーパス一覧](https://sites.google.com/site/shinnosuketakamichi/publication/corpus)
- [無償音声コーパス一覧 (Qiita)](https://qiita.com/nakakq/items/74fea8b55d08032d25f9)
- [音声合成・歌声合成コーパスまとめ (note.com)](https://note.com/npaka/n/na4e7f38d4c1c)
- [個人開発な歌唱DBまとめ](https://km4osm.com/singingvoicedb/)
- [NNSVS レシピ一覧](https://github.com/nnsvs/nnsvs)
