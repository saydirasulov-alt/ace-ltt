# cascade v2.7: RAVC-LTT adaptiv ko'rinish kaskadi (dev tadqiqoti)

> `freeze`, `unseal`, `final` yo'li v2.6.2 bilan bir xil. RAVC faqat dev'da, oldindan ro'yxatdan o'tkazilgan
> bitta ishga tushirish bilan baholanadi. Maqolaning asosiy usuli `view_rule_outcome.txt` bo'yicha
> **edge_LTT**; RAVC natijasi buni o'zgartirmaydi. RAVC qabul qilinsa, overlay + crop CSV/meta fayllarini
> lock/receipt zanjiriga bog'lash alohida qadam.

## Yangi metod: RAVC-LTT

**RAVC-LTT (Risk-certified Adaptive-View Cascade)** eskalatsiya qilingan har bir kadr uchun bitta VLM
chaqirig'ini qiladi. Yorliq va source'ni ishlatmaydigan kichik router (top-box maydoni, boxlar soni,
fire/smoke ballari) `overlay` yoki `crop` ni tanlaydi. Router va chegaralar select fold'da Pareto tartibiga
qo'yiladi; certify fold faqat HB p-qiymatlari va fixed-sequence uchun. Fire/smoke event-risk LTT kafolati
saqlanadi.

## v2.7 o'zgarishlari (prototipga nisbatan)

- `ravc_dev.py`, `ravc_synthetic_validity.py`, `view_bytes.py` ANALYSIS_CODE ichida (codehash ularni qamraydi).
- Bayt narxi kodda yo'q: `protocol.yaml` da `ravc: {byte_cost: ...}` bo'lmasa `ravc_dev` ishlamaydi. Asosiy
  kaskad objective'i (`costs:`) o'zgarmaydi.
- Crop payload: `view_bytes` VLM ko'rgan aynan o'sha crop'ni (`agent_vlm.prepare_image`, max_side 640) JPEG q85
  da o'lchaydi. Eski `nbytes_crop` 448 px edi.
- Operator qatlami: ikki oila. `human` (overlay/crop/RAVC `+human`) — QAROR oilasi, chunki 1/5/20 da
  handoff > 0. `no_human` — diagnostika.
- Manbada fire yoki smoke misoli bo'lmasa, risk NaN bo'ladi (xato emas); manba solishtiruvi faqat chekli
  qiymatlarda.
- Oldindan ro'yxat: `ravc-register` UTC vaqt, protocol/splits/codehash, `ravc` bloki, qabul qoidasi va barcha
  kirish fayllari hash'ini bir marta yozadi. `ravc-dev` birorta hash o'zgargan bo'lsa yoki natija allaqachon
  bor bo'lsa ishlamaydi.
- Sintetik tekshiruv: chegara yaqini (cal 3000) + `--negative-control naive`; haqiqiy risk kalibratsiyada
  ishlatilmagan unitlarda.

## Server tartibi (dev, freeze yo'q)

```bash
STAGE=crop-bytes     NMS=none bash cascade/run_all.sh   # CPU, yorliqsiz
# protocol.yaml ga ravc: {byte_cost: 2.0e-5} (snapshot bilan)
STAGE=ravc-register  NMS=none bash cascade/run_all.sh   # bir marta
TRIALS=100 STAGE=ravc-dev NMS=none bash cascade/run_all.sh
```

MODEL/QUANT standart (3B, none) — mavjud overlay va crop CSV'lari shu model bilan.

## v2.8: ESVA-LTT (Evidence-Stratified Veto Authority) va ceiling diagnostikasi

**Nima uchun.** Dev'da pooled kafolat bajariladi, lekin VLM kichik tutunni "yo'q" deb rad etadi (veto): Pyro'da
kaskad 0.08–0.11, edge 0.02. SmokeBench (Qwen2.5-VL-7B: juda kichik tutunda 0.10, juda kattada 0.69) shu
mexanizmni tasdiqlaydi. RAVC ko'rinish tanlardi — bu muammoga tegmasdi.

**Usul.** Edge detektorning o'zidan yorliqsiz qatlam: S = {eng ishonchli smoke box maydoni <= 0.02 va
s_smoke >= s_fire} (0.02 SmokeBench / PyroNear2025 dan, dev'dan emas). S ichida VLM vetosi lambda ga
kamaytiriladi: g -> min(1, g + lambda). (t_low, t_high, b, a, lambda) bitta LTT oilasi: select fold tartib
beradi (weak Pareto front, tenglarda konservativ oldin), certify fold fixed-sequence. Birgalikdagi IUT:
pooled fire, pooled smoke VA smoke|S. Kafolat: 1-delta ehtimol bilan hammasi <= alpha bir vaqtda.
Sertifikatlash sharti: har risk uchun >= n_min = 45 musbat unit (alpha 0.05, delta 0.10).

**Tartib (avval ceiling, keyin usul):**

```bash
STAGE=diagnose NMS=none bash cascade/run_all.sh          # GO / NO-GO (qoida kod sarlavhasida)
# faqat GO bo'lsa: protocol.yaml ga esva bloki (snapshot bilan), keyin
STAGE=esva-register NMS=none bash cascade/run_all.sh
TRIALS=100 STAGE=esva-dev NMS=none bash cascade/run_all.sh
```

Qabul (ESVA_LTT+human, hammasi): pooled p97.5 <= alpha; sertifikatlash >= 0.90; smoke|S p97.5 <= alpha;
Pyro risk_mean <= 0.05; objective <= 0.90 x edge_LTT. ACCEPT -> taklif qilinadigan usul; REJECT -> edge_LTT.

Sintetik tekshiruv: `python -m cascade.esva_synthetic_validity [--negative-control naive|pooled_only]
[--handoff --cluster 3]`; `pooled_only` S kafolatisiz holatda S riski buzilishini ko'rsatadi.

## v2.9-A: dev probe (diagnostika, yangi inference yo'q)

`STAGE=probe` mavjud ball fayllarini qayta o'qiydi. Hech narsani sertifikatlamaydi va ESVA NO-GO qarorini
o'zgartirmaydi.

- **E. Tartiblash bormi:** agent x manba x (S / S emas) kesimida `g` ning AUC, TPR=0.95 va 0.99 dagi FPR va
  yuqori sezgirlik yo'lagidagi qisman yuza (o'rtacha o'ziga xoslik; 1.0 mukammal, tasodifiy tartiblash ≈0.025).
  Bu kadr darajasidagi tartiblash — hodisa xavfsizligining o'rnini bosmaydi.
- **F. Haqiqiy siyosat fronti:** har bir λ uchun kaskad `run_cascade` bilan bir xil LTT protsedurasida
  kalibrlanadi va ikkinchi yarmida hodisa riski, S riski, FA, calls o'lchanadi. λ=0 bu cascade_LTT; edge_LTT ham
  bir xil bo'linishda qayd etiladi.
- **G. Sertifikatlanuvchanlik:** bitta gipergeometrik hisob emas, haqiqiy bo'linish (dev 50/50, keyin
  calibrate() ning 30/70 select/certify) ko'p marta takrorlanadi. Har bir cheklangan risk uchun certify
  foldidagi musbat unitlar taqsimoti, P(≥ n_min) va 90% ishonch uchun kerakli K chiqadi. Calibration
  rolidagi sonlar yorliq yopiq bo'lgani uchun hisoblanmaydi.
- **H. Sababiy (faqat o'tmish) k-of-n:** vaqt belgili kamera ketma-ketliklarida oxirgi n kadrning k tasi
  chegaradan yuqori bo'lsa signal. Hodisa o'tkazib yuborish, FA va aniqlash kechikishi. Pyro'da 2 dev unit
  borligi uchun bu tavsifiy natija.

```bash
AGENTS="overlay=$OUT/agent_Qwen2.5-VL-3B-Instruct-none-overlay_yolo26-base_s0_nms-none.csv,\
crop=$OUT/agent_Qwen2.5-VL-3B-Instruct-none-crop_yolo26-base_s0_nms-none.csv,\
7b=$OUT/agent_Qwen2.5-VL-7B-Instruct-4bit-overlay_yolo26-base_s0_nms-none.csv" \
PROBE_TRIALS=20 STAGE=probe NMS=none bash cascade/run_all.sh
```

**Statistik doira.** LTT kalibrlash modeli i.i.d. birliklarni faraz qiladi. Almashinuvchanlikning o'zi yetmaydi:
barcha yo'qotishlar bitta Bernoulli o'zgaruvchiga teng bo'lsa ular almashinuvchan, lekin n ortganda axborot
ortmaydi. Shuning uchun maqolada birlik ta'rifi va bog'liqlik farazi asoslanadi, atama almashtirilmaydi.
n=17 da sertifikatlashning imkonsizligi ham "har qanday test" uchun emas: taqsimotdan mustaqil, deterministik
va yo'qotish kamayganda sertifikatlashni qiyinlashtirmaydigan testlar sinfi uchun.

## v2.10: payload byudjeti va yagona o'lchov yo'li

**Tuzatilgan kamchilik.** v2.9 da baytlar JPEG q85 da hisoblanardi, model esa siqilmagan tasvirni ko'rardi.
Endi `agent_vlm --jpeg-quality Q` bitta yo'lni beradi: tayyorlash -> kodlash -> baytlarni sanash -> ochish ->
model aynan shu baytlarni ko'radi. Baytlar agent CSV'ning `nbytes_payload` ustuniga yoziladi.
Payload ta'rifi: crop uchun JPEG(crop); overlay uchun JPEG(kadr) + box metadata baytlari, boxlar bulutda,
ochilgandan keyin chiziladi. Tarmoq sarlavhalari va qayta uzatishlar o'lchanmaydi, shuning uchun da'vo faqat
application payload haqida. `--jpeg-quality 0` eski yo'l bo'lib qoladi va payload o'lchanmaydi.

**Byudjet ta'rifi** (`payload_budget.py`): mustaqil birlik u uchun C_u = birlikdagi baholanadigan kadrlarga
to'g'ri keladigan o'rtacha uzatilgan bayt; yo'qotish 1{C_u > B0}; risk R_B = P_U(C_U > B0) <= beta.
Bu kadr, kamera yoki soniya bo'yicha limit emas va qisqa portlashlarni taqiqlamaydi. Bitrate kafolati emas.

**Sertifikatlash**: p_joint = max(p_fire, p_smoke, p_B), Pareto testing ichida. Nomzodlar oilasiga siyosat
(ko'rinish, lambda) va B0 ning o'zi ham kiradi, shuning uchun bir nechta byudjet uchun egri chiziq bir vaqtda
haqiqiy bo'ladi. Mexanizm yangi emas: Pareto Testing bir nechta cheklovni allaqachon birga sertifikatlaydi,
quantile LTT esa oshib ketish ehtimolini nazorat qiladi. Tekshiriladigan narsa — ko'rinish tanlovi va selektiv
veto xavf/FA/payload nisbatini fixed-crop LTT va bir xil uch cheklovli standart Pareto-LTT'ga nisbatan
yaxshilaydimi.

## v2.11: yig'ish (acquisition) narvoni — yopilgan tajriba ro'yxati

Metod taxmin qilib qurilmaydi; avval nima ko'rsatilishi bo'yicha beshta variant o'lchanadi. Ro'yxat
`study_register` bilan yopiladi va qaror qoidasi `cascade/study_register.py` ichida yozilgan.

| kod | buyruq qismi | nima tekshiriladi |
|---|---|---|
| A | `--view crop --jpeg-quality 85` | mos o'lchov yo'li (ma'lumotnoma) |
| B | `--view overlay --jpeg-quality 85` | to'liq kadr |
| C | `--view crop --zoom 2 --jpeg-quality 85` | ko'rinadigan tutun maydonini kattalashtirish |
| D | `--view crop --crop-pad 3 --jpeg-quality 85` | kengroq kontekst |
| E | `--view crop --pair prev --jpeg-quality 85` | vaqt jufti (oldingi kadr yonma-yon), faqat vaqt belgili manbalar |

Qaror qoidasi: variant S ichida AUC ni ≥0.05 oshirsa yoki FPR@TPR95 ni ≥0.10 tushirsa **va** bu uchala
manbada alohida takrorlansa **va** farqning 90% oralig'i (birlik darajasidagi bootstrap) nolni o'z ichiga
olmasa — GO. Kadr darajasidagi xatolik ishlatilmaydi: S ichidagi 2951 musbat kadrning 2570 tasi ikkita Pyro
kamerasidan. Pyro'dagi yaxshilanish umumlashtirilmaydi. Hech biri o'tmasa, yig'ish o'qi yopiladi va manfiy
natija shu holicha yoziladi.

`dev_probe` ning I bo'limi shu taqqoslashni birlik bootstrapi bilan beradi (`--reference`, `--boot`).

## Kafolat nima deydi va nima demaydi

**Siyosat.** θ = (t_low, t_high, b, a):

- `s ≥ t_high` bo'lsa — signal;
- `s < t_low` bo'lsa — sukut;
- oraliqda kadr VLM agentga yuboriladi:
  - `g ≥ a` — signal;
  - `g < b` — rad;
  - oraliqda — operatorga.

`s` — detektor balli. `g` = sigmoid(logit_Yes − logit_No) — **tartiblash balli**, kalibrlangan ehtimol emas.

**Learn-then-Test bayonoti** (kalibrlash tanlanmasi ustida):

> P( haqiqiy xavfi α dan katta bo'lgan konfiguratsiya sertifikatlanadi ) ≤ δ,

bu kalibrlash va kelajakdagi birliklar (kamera / video / dublikat guruhi) almashinuvchan bo'lganda o'rinli.

- **Cheklangan xavflar:** fire hodisasini o'tkazib yuborish va smoke hodisasini o'tkazib yuborish.
- **Qo'shma test:** p = max(p_fire, p_smoke), ya'ni intersection–union testi.
- **Pareto Testing:** grid va Pareto tartibi faqat calibration'ning 30% "select" qismida quriladi; qolgan 70% faqat p-qiymat hisoblaydi. Bonferroni esa ma'lumotdan mustaqil grid ishlatadi.

**Bu kafolat quyidagini aytmaydi:** "cheklangan testdagi empirik xavf α dan oshgan sinovlar ulushi ≤ δ". Hisobotda quyidagilar beriladi:

- `certification_rate`;
- `cert_and_test_exceeds_alpha` — faqat diagnostika;
- empirik xavf: sinovlar bo'yicha oraliq; final'da esa 5000 takrorli birlik (klaster) bootstrap CI — umumiy (max) xavf uchun, shuningdek fire va smoke uchun alohida.

δ bilan solishtirish faqat `--synthetic-validity` da bajariladi, chunki u yerda haqiqiy xavf ma'lum. Verdikt qoidasi: Clopper–Pearson yuqori chegarasi ≤ δ bo'lsa **supported**, pastki chegarasi > δ bo'lsa **VIOLATED**, aks holda **inconclusive**.

**Sintetik natija** (60 ming birlikli populyatsiya, 1500 kalibrlash birligi, 200 sinov, α = 0.05, δ = 0.1). Jadvalda P(sertifikatlangan va haqiqiy xavf > α) va o'rtacha haqiqiy xavf:

| usul | frame: P | frame: xavf | event: P | event: xavf |
|---|---|---|---|---|
| `cascade_LTT` | 0.000 | 0.031 | 0.000 | 0.029 |
| `cascade_LTT_bonf` | 0.000 | 0.018 | 0.000 | 0.019 |
| `edge_LTT` | 0.000 | 0.027 | 0.000 | 0.028 |
| `cascade_LTT_image` | 0.050 | 0.044 | 0.000 | 0.005 |
| `edge_LTT_image` | 0.020 | 0.039 | 0.000 | 0.005 |

Hammasi "supported". Rasm darajasidagi variantlar kafolatni buzmaydi, lekin "event" maqsadida haddan tashqari ehtiyotkor (xavf 0.005 ≪ α), ya'ni qimmatroq. Bu raqamlar faqat sintetik va maqolaga natija sifatida kirmaydi; real ma'lumotdagi narx farqi alohida tekshiriladi.

**"Event" xavfi — klaster bo'yicha o'rtacha hodisa xavfi.** Ko'p kadrli hodisa faqat Pyro-SDIS'da bor. D-Fire, FASDD va FLAME2 kadrlari bittalik hodisalar, ya'ni u yerda rasm xavfining o'zi.

## Ish tartibi (o'zgarmas)

```
protocol → heads → scores → records → dev → [protocol.yaml yakunlanadi] → freeze
         → records (qayta) → validity (faqat diagnostika) → unseal (bir marta) → final (bir marta)
```

`validity` lock bo'lmasa, yoki records lock'dan oldin qurilgan bo'lsa, **rad etiladi**. Freeze'dan keyin hech narsa o'zgartirilmaydi — **tahlil kodi ham** (quyida `analysis_code_sha256`).

## Birlik (unit) va rollar

**Birlik = `split_group`.** Bu fsclean dublikat grafining **barcha manbalar bo'yicha birga** qurilgan global komponenti. Manbalararo pikselda tasdiqlangan dublikat klasteri ataylab **bitta** birlik hisoblanadi.

Buni v0.2 da tekshirdim: test'dagi 17 ta ko'p manbali guruhdan 16 tasida bitta `dup_cluster` ikkala manbani qamrab oladi. Masalan, g354: 38 D-Fire + 200 FASDD kadri — bir xil rasmlar. Uni `source:split_group` ga bo'lish bu rasmlarni ikki "mustaqil" birlikka ajratib, calibration va sealed o'rtasida sizib chiqish hosil qilardi.

**Audit jadvali** (`STAGE=audit`, yorliqsiz; `init` ham oxirida chiqaradi; `<P>/unit_audit.csv` ga yoziladi). v0.2 dagi natija:

| rol | ko'p manbali birliklar | izoh |
|---|---|---|
| calibration | 11 | har biri bitta birlik, asosiy manbasi qatlamida tortilgan |
| sealed_test | 6 | xuddi shunday |
| rollararo birlik | 0 | `init` da assert bilan tekshiriladi |

Serverdagi `STAGE=audit` chiqishi (birliklar soni va ulardagi kadrlar) maqoladagi jadval uchun manba bo'ladi.

**Stratifikatsiya.** Har bir test guruhi faqat bitta qatlamga tushadi — o'zining asosiy manbasiga (eng ko'p kadrli manba). Shu tufayli hech bir guruh ikki marta tortilmaydi. v2.2 dagi birlashtirish xatosi tuzatildi: haqiqiy v0.2 da ko'p manbali birliklar endi 11 calibration / 6 sealed bo'ldi (ilgari 14 / 3 edi).

**Kichik manbalar.** Test birliklari `min_test_units_per_source` (5) tadan kam bo'lgan manba `external_shift` ga o'tkaziladi. v0.2 da bu **FLAME2** (test'da 1 video). Uning natijasi siljish sifatida beriladi, kafolat da'vo qilinmaydi.

**Pyro-SDIS** chegarada: calibration'da 2 kamera (94 hodisa), sealed'da 3 kamera. Bu cheklov sifatida yoziladi.

**Yorliqlarni yashirish:**

- freeze'dan oldin records'da **faqat dev** yorliqlari bo'ladi;
- freeze'dan keyin **dev + calibration**;
- `--unseal` bilan hammasi, shu jumladan sealed_test va external_shift.

| rol | manba | vazifa |
|---|---|---|
| dev | val | head (`head_rule`), VLM, view, narxlar, ixtiyoriy grid tanlanadi |
| calibration | test birliklarining 50% i, manba bo'yicha stratifikatsiya | faqat kalibrlash |
| sealed_test | test birliklarining qolgan 50% i | yorliqlari lock'gacha records'ga kirmaydi |
| external_shift | tuman/xira tashqi to'plam | siljish sinovi, kafolat da'vo qilinmaydi |

## Server: o'rnatish (eski papkani o'chirmasdan)

```bash
cd ~/PROJECT/lha-yolo26 && source ~/PROJECT/activate.sh
mv cascade cascade_backup_$(date +%Y%m%d_%H%M)             # eski versiya kelib chiqish dalili sifatida saqlanadi
mkdir -p ~/PROJECT/staging/cascade_v2_6_2 && cd ~/PROJECT/staging/cascade_v2_6_2
python -m zipfile -e ~/PROJECT/cascade_v2_6_2.zip .
STAGE=selftest bash cascade/run_all.sh                       # oxirida "N/N checks passed" va "SELFTEST OK"
cp -r cascade ~/PROJECT/lha-yolo26/ && cd ~/PROJECT/lha-yolo26
```

## Bosqichlar

```bash
STAGE=protocol bash cascade/run_all.sh
STAGE=heads    bash cascade/run_all.sh     # SELECTED qatori -> protocol.yaml: detector.nms
NMS=<tanlangan> STAGE=scores  bash cascade/run_all.sh
NMS=<tanlangan> STAGE=records bash cascade/run_all.sh
NMS=<tanlangan> STAGE=dev     bash cascade/run_all.sh
# protocol.yaml ni yakunlang (alpha, delta, costs, fixed_baselines, ...). Shundan keyin o'zgartirilmaydi.
NMS=<tanlangan> STAGE=freeze   bash cascade/run_all.sh
NMS=<tanlangan> STAGE=records  bash cascade/run_all.sh
NMS=<tanlangan> STAGE=validity bash cascade/run_all.sh
NMS=<tanlangan> STAGE=unseal   bash cascade/run_all.sh   # BIR MARTA: receipt -> ochilgan records -> hash
NMS=<tanlangan> STAGE=final    bash cascade/run_all.sh   # BIR MARTA, faqat receipt'dagi records hash'i bilan
# uzilib qolsa (faqat shunda): RESUME=1 NMS=<tanlangan> STAGE=unseal|final bash cascade/run_all.sh
STAGE=verify-final bash cascade/run_all.sh # faqat o'qiydi: final.md, final.json, records <-> receipt; lock + kod
STAGE=figures-dev  bash cascade/run_all.sh # dev figuralari (TikZ), istalgan vaqtda dev'dan keyin
STAGE=audit    bash cascade/run_all.sh     # istalgan vaqtda, yorliqsiz
STAGE=codehash bash cascade/run_all.sh     # o'rnatilgan kodning analysis_code_sha256 qiymati
```

## Figuralar (TikZ / pgfplots)

Har bir figura ikki fayldan iborat:
- `figN_*.tex` — bitta mustaqil `tikzpicture` (pgfplots, ma'lumotlar ichida, tashqi fayl yo'q);
- `figN_*.csv` — chizilgan raqamlar.

Maqolada `\input{figN_*.tex}` ishlatiladi; kerakli paketlar `figure_preamble.tex` da. `pdflatex` bo'lsa, aynan shu `.tex` dan `preview/*.pdf` va `*.png` kompilyatsiya qilinadi — bu faqat tekshirish uchun. LaTeX bo'lmasa `.tex` + `.csv` baribir yoziladi, xato chiqmaydi. Tekshiruv PDF'i Times shrifti bilan, u bo'lmasa (masalan, serverda shrift fayli yetishmasa) standart shrift bilan kompilyatsiya qilinadi; bu `figures_manifest.json` ga yoziladi. Xato faqat `.tex` ning o'zi kompilyatsiya bo'lmasa chiqadi.

| figura | manba | main / supplement |
|---|---|---|
| fig1_architecture | ma'lumotsiz sxema (edge → cloud VLM → operator, `t_low`, `t_high`, `b`, `a`) | main |
| fig2_dataset_audit | `dataset_audit.json` (yorliqsiz): tozalashdan oldin/keyin, asl split'lardagi leakage, split_group o'lchamlari, pHash sezgirligi | main |
| fig3_risk_cost | event miss risk ↔ bulut chaqiruvlari / uplink / kechikish; sertifikatlangan = to'la marker | main |
| fig4_forest | fire, smoke va max risk + 95% bootstrap CI (final), α chizig'i | main |
| fig5_edge_cloud_cost | bulut chaqiruvlari %, uplink KB/kadr, kechikish | main |
| fig6_by_source | manbalar bo'yicha risk (TikZ issiqlik xaritasi) + external shift (kafolatsiz) | main |
| fig7_head_selection | dev'da head tanlovi (recall, negativ eskalatsiya, mAP50, kechikish) | supplement |

**`STAGE=figures-dev`** faqat `dev_*` fayllarni, `head_report.json` va `dataset_audit.json` ni o'qiydi. fig3–fig6 ning dev nusxalarida "DEV (val role)" belgisi bor — ular maqola natijasi emas.

**`STAGE=figures-final`** faqat `final.json` ni o'qiydi. Undan oldin tekshiradi:
- receipt holati `done`;
- `final.json` va `final.md` hash'lari receipt bilan bir xil;
- lock va kod hash'i mos.

Records, yorliqlar va ballar ochilmaydi, receipt o'zgartirilmaydi.

**Final bilan bog'liqlik:**
- `final` tugagach figuralar avtomatik chiziladi.
- Figurada xato bo'lsa, final **qayta bajarilmaydi**: kod tuzatiladi va `STAGE=figures-final` ishga tushiriladi.
- Freeze'dan keyin **faqat** `figures.py` / `figure_style.py` o'zgargan bo'lsa, `FIGFIX=1 STAGE=figures-final` ruxsat beradi. O'zgargan fayllar `figures_manifest.json` ga yoziladi va maqolada aytiladi. Boshqa har qanday kod o'zgarishi rad etiladi.
- `final.json` endi `results`, `by_source`, `external_shift` va `method_order` ni saqlaydi, shuning uchun figura kodi Markdown'ni parse qilmaydi.

**Oldindan belgilangan spetsifikatsiya** (`figure_style.py` → `DEFAULT_SPEC`, kod hash'iga kiradi):
- metodlar tartibi va main/supplement bo'linishi;
- o'q chegaralari: risk ≤ 0.20, uplink ≤ 400 KB, kechikish ≤ 5000 ms.

Freeze'dan oldin `protocol.yaml` dagi `figures:` bloki orqali o'zgartirish mumkin. Chegaradan chiqqan qiymat chegarada boshqa marker bilan chiziladi va CSV'da `clipped` deb belgilanadi.

**Kelishilgan bo'linish** (`protocol.yaml` → `figures:`, freeze bilan qulflanadi):

```yaml
figures:
  main: [fig1_architecture, fig2_dataset_audit, fig3_risk_cost, fig4_forest, fig6_by_source]
  supplement: [fig5_edge_cloud_cost, fig7_head_selection]
```

**Uslub:** ranglar Okabe–Ito (rang ko'rligiga mos), issiqlik xaritasi cividis. Kengliklar `\textwidth` / `\columnwidth` ga nisbatan, matn `\footnotesize` (IEEEtran'da 8 pt).

**Dataset audit (fig 2)** PC'da ishga tushiriladi, chunki `clean_work` shu yerda. `cascade\` papkasi turgan joyda, Anaconda Prompt'da:

```bat
python -m cascade.dataset_audit --dataset <DATASET_ROOT>/FireSmoke-Clean_v0.2 --work <DATASET_ROOT>/clean_work --out dataset_audit.json
```

Keyin `dataset_audit.json` ni serverdagi `~/PROJECT/runs/cascade/` ga nusxalang.

**Arxivlangan audit (v0.2):** paketdagi modul bilan ikki marta qayta yaratilgan, ikkalasi baytma-bayt bir xil. SHA-256: `ab4732bf599358c23bc180d3e37cd178df921234d8c8cbb4bb224b21c9595d0b`.

Asosiy raqamlar:
- 61,284 split_group, ulardan 188 tasi ko'p manbali;
- split'lar orasidan o'tgan guruh: 0;
- D-Fire test, `t=8`, pikselda tasdiqlangan qo'shni: 51.30%.

**Maqola matni uchun to'g'ri ifoda:**
- "FireSmoke-Clean test'da pikselda tasdiqlangan qo'shni **pikselda tekshirish mavjud bo'lgan barcha `t ≤ 10` chegaralarida** 0%".
- `t = 14` dagi 95.13% faqat hash nomzodlari (look-alike), pikselda tasdiqlanmagan.

## Head tanlash qoidasi (`head_rule`, deterministik, faqat dev)

1. **Mos keluvchi head:** fire va smoke hodisalari bo'yicha `s_min` dagi candidate recall'i har biri eng yaxshi head'dan ≤ 1 pp past bo'lishi kerak.
2. **Mos keluvchilar orasida:** eng past negativ eskalatsiya tanlanadi. Undan ≤ 1 pp farqlilar ichida — yuqori dev mAP50. mAP50 ≤ 0.5 pp farq qilsa — past kechikish. Hali ham teng bo'lsa — `nms "false"`.
3. **Hech biri mos kelmasa:** min(fire, smoke) recall eng katta bo'lgan head.

mAP50 aynan deploy parametrlarida hisoblanadi: imgsz 640, conf 0.001, iou 0.7, max_det 100, meta'dagi nms.

## Himoyalar (hammasi `selftest` da tekshiriladi)

**Selftest qabul mezoni:**
- exit code 0;
- birorta ham `FAIL` qatori yo'q;
- oxirgi qatorlar `N/N checks passed` va `SELFTEST OK`.

Tekshiruvlar soni muhitga bog'liq. Masalan, TikZ kompilyatsiya tekshiruvi `pdflatex` bo'lmasa `[SKIP]` bo'ladi (`.tex`/`.csv` baribir tekshiriladi): v2.6.2 da pdflatex bilan 85/85, pdflatex'siz 84/84 + 1 skipped. `run_all.sh` ning o'zi ham tekshiriladi: toza muhitda (`OUT`/`PROTO`/`AUDIT` eksport qilinmagan) `set -u` ostida ishga tushishi va barcha STAGE nomlari mavjudligi.

**Lock quyidagilarni qamrab oladi:** protocol.yaml, splits.csv, **detektor CSV va agent CSV mazmuni**, ikkala meta-fayl, (ishlatilsa) plain-agent CSV va meta, hamda **`analysis_code_sha256`**. Ulardan birortasi o'zgarsa:

- `build_records` to'xtaydi;
- `--unseal` rad etiladi;
- `validity` va `final` rad etiladi.

**`analysis_code_sha256`.** Tahlilga ta'sir qiluvchi fayllar: `__init__.py`, `dataio.py`, `risk.py`, `policy.py`, `select.py`, `experiment.py`, `build_records.py`, `run_cascade.py`, `make_protocol.py`, hamda figura kodi: `dataset_audit.py`, `figure_style.py`, `figures.py`. Hash qat'iy tartibda (fayl nomi + xom baytlarining sha256) hisoblanadi. `freeze` uni fayl-bo'yicha hash'lar bilan birga lock'ga yozadi va lock digest'iga qo'shadi. Keyin har bir `build_records`, `validity`, `--unseal` va `final` joriy kodni qayta hisoblaydi; farq bo'lsa, qaysi fayl o'zgarganini aytib rad etadi.

- `dump_detector.py` va `agent_vlm.py` kiritilmagan: ularning natijasi lock'dagi CSV + meta mazmuni orqali qamralgan.
- `head_report.py`, `selftest.py`, `simulate.py` freeze'dan keyin ishlatilmaydi.
- Selftest paketning **nusxasida** `policy.py` ning bitta baytini o'zgartiradi: records, validity, `--unseal` (receipt ham yaratilmaydi) va final rad etiladi. Asl paket o'zgarmaydi.
- v2.3 bilan qilingan lock'da bu maydon yo'q, shuning uchun u yaroqsiz hisoblanadi. Serverda hali freeze qilinmagan, bu muammo emas.

**`freeze` quyidagilarni tekshiradi:**
- manifest hash;
- agent aynan shu detektor CSV'dan olinganini (mazmun hash'i);
- detektor va agent sozlamalari protokolga mosligini;
- `--dry` agent emasligini.

**`freeze` qo'shimcha ravishda:** `nms "false"` bo'lsa runtime `end2end` true bo'lishi, boshqa holatda false bo'lishi shart.

**Protokol receipt'i** (`<P>/final_receipt.json`) — bir tomonlama holat mashinasi:

```
none → unsealing → unsealed(records_sha256) → running → done
```

- Receipt **to'liq va atomik** yaratiladi: JSON avval vaqtinchalik faylga yoziladi va `fsync` qilinadi, keyin receipt nomiga hard-link qilinadi. Nom band bo'lsa, link xato beradi. Shuning uchun uzilishda yo receipt umuman bo'lmaydi, yo u to'liq bo'ladi — yarim yozilgan JSON qolmaydi. Keyingi yangilanishlar ham temp + `fsync` + `os.replace` orqali. Hard-link'ni qo'llamaydigan fayl tizimida `O_EXCL` + `fsync` ishlatiladi. Buzilgan JSON bo'lsa, rad etiladi (fail-closed).
- `--unseal` tartibi: manifest **faylining** hash'i va lock (kod hash'i bilan) tekshiriladi → receipt `O_EXCL` bilan atomik yaratiladi (`unsealing`) → **shundan keyingina** `manifest.csv` (yorliqlar shu faylda) o'qiladi, talqin qilinadi va yoziladi → records sha256 i receipt'ga yoziladi (`unsealed`). Ya'ni receipt manifest xotiraga o'qilishidan oldin mavjud bo'ladi.
- Ikkinchi `--unseal` (istalgan fayl nomiga) rad etiladi. Sealed build receipt'dagi records faylini ustidan yoza olmaydi.
- `final` faqat hash'i receipt'dagi `records_sha256` ga teng bo'lgan records'ni qabul qiladi. Ichki jihatdan izchil (meta'si mos) boshqa records ham rad etiladi. Boshlanishda holat `running` ga, oxirida `done` ga o'tadi va `final_md_sha256` hamda `final_json_sha256` yoziladi.
- **Uzilish:** receipt o'chirilmaydi. `RESUME=1` bilan:
  - `unseal` — faqat `unsealing`/`unsealed` holatda, aynan shu `--out` va lock bilan. Receipt'da hash allaqachon bo'lsa, qayta qurilgan fayl aynan shu hash'ni berishi shart.
  - `final` — faqat `running` holatda, aynan shu `--out-dir` bilan. Yarim qolgan `final.md`/`final.json` o'chirilmaydi, `*.crashed_N` deb qayta nomlanadi va sha256 i receipt'ga yoziladi.
- Har bir voqea (`unseal_start`, `unseal_resume`, `unsealed`, `final_start`, `final_resume`, `done`) vaqt belgisi bilan receipt'da qoladi. `final.md` sarlavhasida resume'lar soni chiqadi; ular maqolada aytiladi.

**`final` qo'shimcha ravishda tekshiradi:**
- lock (kod hash'i bilan) va records meta'sidagi barcha hash'larni;
- records `--unseal` bilan qurilganini;
- barcha UID'lar mavjudligini va har bir rolda yorliqlar to'liqligini.

## Baseline'lar

- `edge_fixed`, `cascade_fixed`, `detector_gated_agent` — chegaralari `protocol.yaml` → `fixed_baselines` dan olinadi.
- `edge_LTT_image` — rasm darajasida almashinuvchan deb olingan HB/LTT (conformal usul emas).
- `all_frames_cloud_overlay` — overlay agent har bir kadrda. `run_all.sh` da **ulanmagan** va protokolda `false` qoladi. Sabab: asosiy agent `s_min` bilan ishlaydi, `true` qilinsa `run_cascade` barcha kadrlar uchun `g` talab qilib rad etadi. Detektordan mustaqil bulut baseline sifatida `pure_cloud_plain` ishlatiladi.
- `pure_cloud_plain` — detektordan mustaqil sof bulut baseline. Ishga tushirish: `PURE_CLOUD=1` va `pure_cloud_baseline: true`. `run_all.sh` plain agentni (`--view plain --s-min 0`) ishga tushiradi va `--agent-plain` ni freeze, records va final bosqichlariga uzatadi. Kechikishi = plain agent vaqti, detektor vaqti 0.

## Platforma

Server (Linux) asosiy muhit. Protokoldagi yo'llar JSON tarzida qo'shtirnoqqa olinadi, shuning uchun Windows'dagi `C:\...` yo'llari YAML'ni buzmaydi. Testlar fayllarni baytma-bayt solishtiradi.

## Halollik qoidalari

- `--dry`, `simulate` va `selftest` raqamlari maqolaga kirmaydi.
- `agent_ms` batch bo'yicha o'rtacha; real kechikish real qurilmada o'lchanadi.
- Operatorga o'tkazilgan kadr to'g'ri hal qilingan deb olinadi; bu faraz maqolada yozilsin.
- Pyro-SDIS test'da atigi 5 kamera bor; bu cheklov sifatida yozilsin.
- Test yorliqlarini model yordamida tuzatmang.
