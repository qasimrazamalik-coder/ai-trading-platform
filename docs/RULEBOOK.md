# Rulebook

Every hard rule the bot enforces, with its source and the code that enforces it.
Generated from `packages/core/src/quant_os_core/ops/rulebook.py` (`rulebook.markdown()`);
a test fails when this file and the code drift apart. The dashboard shows the same list
read-only. Changing a rule needs a code change, a test, a reason in `CHANGELOG.md` and,
for Shariah rules, a scholar's written sign-off (see `GUARDRAILS.md`).

## Shariah

### SH-01

- English: Spot ownership only. Futures, forwards, options, binary options, CFDs, perpetuals, index products and prediction markets are blocked (R1, R2).
- اردو: صرف اسپاٹ ملکیت۔ فیوچرز، فارورڈز، آپشنز، بائنری آپشنز، سی ایف ڈی، پرپیچوئلز، انڈیکس پروڈکٹس اور پیشگوئی کی منڈیاں بند ہیں (R1، R2)۔
- العربية: التملّك الفوري فقط. العقود الآجلة والمستقبلية والخيارات والخيارات الثنائية وعقود الفروقات والعقود الدائمة ومنتجات المؤشرات وأسواق التوقعات محظورة (R1، R2).
- Source: GUARDRAILS.md rule 1; IIFA Resolution 63
- Enforced by: `quant_os_core.shariah.rules:ShariahGate.check_instrument`, `quant_os_core.shariah.rules:BLOCKED_INSTRUMENT_TYPES`, `quant_os_core.shariah.rules:BLOCKED_ASSET_CLASSES`

### SH-02

- English: No margin account and no buying above free cash: leverage is not permitted (R3).
- اردو: مارجن اکاؤنٹ نہیں اور دستیاب نقد سے زیادہ خریداری نہیں: لیوریج کی اجازت نہیں (R3)۔
- العربية: لا حساب هامش ولا شراء بأكثر من النقد المتاح: الرافعة المالية غير مسموحة (R3).
- Source: GUARDRAILS.md rules 2 and 8; IIFA Resolution 63
- Enforced by: `quant_os_core.shariah.rules:ShariahGate.check_order`

### SH-03

- English: No interest, financing or swap charges: positions that carry them are blocked (R5).
- اردو: سود، فنانسنگ یا سواپ چارجز نہیں: ایسی پوزیشنیں بند ہیں (R5)۔
- العربية: لا فوائد ولا رسوم تمويل أو مبادلة: المراكز التي تحملها محظورة (R5).
- Source: GUARDRAILS.md rule 2
- Enforced by: `quant_os_core.shariah.rules:ShariahGate.check_order`

### SH-04

- English: No short selling: a sell larger than the quantity owned is refused (R4).
- اردو: شارٹ سیلنگ نہیں: ملکیت سے زیادہ فروخت رد کر دی جاتی ہے (R4)۔
- العربية: لا بيع على المكشوف: يُرفض أي بيع يتجاوز الكمية المملوكة (R4).
- Source: GUARDRAILS.md rule 3; IIFA Resolution 63
- Enforced by: `quant_os_core.shariah.rules:ShariahGate.check_order`, `quant_os_core.risk_engine.engine:RiskEngine.evaluate`, `quant_os_core.shariah.rules:halal_risk_limits`

### SH-05

- English: Shares and funds must pass the AAOIFI ratios: interest-bearing debt and interest-bearing cash each under 30% of market value, non-compliant revenue under 5%, tangible assets at least one third (R6).
- اردو: شیئرز اور فنڈز کو AAOIFI کے تناسب پاس کرنا ہوں گے: سودی قرض اور سودی نقد ہر ایک مارکیٹ ویلیو کے 30% سے کم، غیر شرعی آمدنی 5% سے کم، ٹھوس اثاثے کم از کم ایک تہائی (R6)۔
- العربية: يجب أن تجتاز الأسهم والصناديق نسب أيوفي: الديون الربوية والنقد الربوي كلٌّ منهما أقل من 30% من القيمة السوقية، والإيرادات غير المتوافقة أقل من 5%، والأصول الملموسة ثلث على الأقل (R6).
- Source: GUARDRAILS.md rule 4; AAOIFI Shariah Standard 21
- Enforced by: `quant_os_core.shariah.rules:ShariahGate.screen_company`, `quant_os_core.shariah.rules:ShariahProfile`

### SH-06

- English: Missing screening data blocks the buy: without a fund's or company's financials the bot cannot verify compliance (R6).
- اردو: اسکریننگ ڈیٹا نہ ہو تو خریداری بند: فنڈ یا کمپنی کے مالیاتی اعداد کے بغیر بوٹ شرعی مطابقت کی تصدیق نہیں کر سکتا (R6)۔
- العربية: غياب بيانات الفحص يمنع الشراء: دون البيانات المالية للصندوق أو الشركة لا يمكن التحقق من التوافق (R6).
- Source: GUARDRAILS.md rule 4; AAOIFI Shariah Standard 21
- Enforced by: `quant_os_core.shariah.rules:ShariahGate.check_order`

### SH-07

- English: Sector screen: alcohol, pork, conventional finance and insurance, gambling, adult entertainment and tobacco are excluded (R7).
- اردو: شعبہ جاتی اسکرین: شراب، خنزیر، روایتی مالیات اور انشورنس، جوا، فحش تفریح اور تمباکو خارج ہیں (R7)۔
- العربية: فحص القطاع: الخمور ولحم الخنزير والتمويل والتأمين التقليديان والقمار والترفيه الإباحي والتبغ مستبعدة (R7).
- Source: GUARDRAILS.md rule 4; AAOIFI Shariah Standard 21
- Enforced by: `quant_os_core.shariah.rules:ShariahGate.screen_company`, `quant_os_core.shariah.rules:CORE_EXCLUDED_SECTORS`

### SH-08

- English: Crypto only under a profile that enables it, and only tokens that pass the token screen; a missing token profile blocks the buy (R13).
- اردو: کرپٹو صرف اسی پروفائل میں جو اسے فعال کرے، اور صرف وہ ٹوکن جو ٹوکن اسکرین پاس کریں؛ ٹوکن پروفائل نہ ہو تو خریداری بند (R13)۔
- العربية: العملات المشفّرة فقط ضمن ملف يسمح بها، وفقط الرموز التي تجتاز فحص الرموز؛ وغياب ملف الرمز يمنع الشراء (R13).
- Source: GUARDRAILS.md rule 5
- Enforced by: `quant_os_core.shariah.rules:ShariahGate.screen_crypto`, `quant_os_core.shariah.rules:ShariahGate.check_instrument`, `quant_os_core.shariah.rules:ShariahGate.check_order`

### SH-09

- English: Speculative forex is blocked (R10).
- اردو: سٹے بازی والا فاریکس بند ہے (R10)۔
- العربية: تداول العملات المضاربي محظور (R10).
- Source: GUARDRAILS.md rule 6; AAOIFI Shariah Standard 1
- Enforced by: `quant_os_core.shariah.rules:ShariahGate.check_instrument`

### SH-10

- English: Venues not licensed in the owner's jurisdiction are blocked; Pakistani residents may not use offshore forex or CFD venues (R14).
- اردو: مالک کے دائرہ اختیار میں غیر لائسنس یافتہ پلیٹ فارم بند ہیں؛ پاکستانی رہائشی آف شور فاریکس یا سی ایف ڈی پلیٹ فارم استعمال نہیں کر سکتے (R14)۔
- العربية: المنصات غير المرخّصة في بلد المالك محظورة؛ ولا يجوز للمقيمين في باكستان استخدام منصات الفوركس أو عقود الفروقات الخارجية (R14).
- Source: GUARDRAILS.md rule 6
- Enforced by: `quant_os_core.shariah.rules:ShariahGate.check_order`

### SH-11

- English: Selling a position you own, to exit, is always allowed, even if the asset later fails the screen.
- اردو: اپنی ملکیتی پوزیشن بیچ کر نکلنا ہمیشہ جائز ہے، چاہے وہ اثاثہ بعد میں اسکرین میں ناکام ہو جائے۔
- العربية: بيع مركز مملوك للخروج مسموح دائمًا، حتى لو أخفق الأصل في الفحص لاحقًا.
- Source: GUARDRAILS.md rule 7
- Enforced by: `quant_os_core.shariah.rules:ShariahGate.check_order`

## Risk

### RK-01

- English: Leverage is fixed at 1.0; it cannot be raised by configuration.
- اردو: لیوریج 1.0 پر مقرر ہے؛ کنفیگریشن سے اسے بڑھایا نہیں جا سکتا۔
- العربية: الرافعة المالية ثابتة عند 1.0 ولا يمكن رفعها من الإعدادات.
- Source: GUARDRAILS.md rule 8; risk engine
- Enforced by: `quant_os_core.shariah.rules:halal_risk_limits`, `quant_os_core.risk_engine.engine:RiskEngine.evaluate`

### RK-02

- English: Every entry has a stop below the entry, a target above it and a written invalidation condition before it is sent.
- اردو: ہر انٹری سے پہلے انٹری سے نیچے اسٹاپ، اوپر ہدف اور لکھی ہوئی منسوخی کی شرط لازمی ہے۔
- العربية: لكل دخول وقف أدنى من سعر الدخول وهدف أعلى منه وشرط إلغاء مكتوب قبل إرساله.
- Source: GUARDRAILS.md rule 9
- Enforced by: `quant_os_core.agents.trade_plan:TradePlan.check_geometry`

### RK-03

- English: Risk per trade at most 1% of equity and position at most 20% (balanced dial); 0.5% and 15% on conservative; the Growth dial is opt-in and never above 2% and 30%. The risk engine's halal limits also cap any position at 20% of equity.
- اردو: فی ٹریڈ خطرہ زیادہ سے زیادہ ایکویٹی کا 1% اور پوزیشن زیادہ سے زیادہ 20% (متوازن ڈائل)؛ محتاط پر 0.5% اور 15%؛ گروتھ ڈائل اختیاری ہے اور کبھی 2% اور 30% سے زیادہ نہیں۔ رسک انجن کی حلال حدیں بھی ہر پوزیشن کو ایکویٹی کے 20% تک محدود رکھتی ہیں۔
- العربية: المخاطرة لكل صفقة 1% من رأس المال كحد أقصى والمركز 20% كحد أقصى (الوضع المتوازن)؛ و0.5% و15% في الوضع المحافظ؛ ووضع النمو اختياري ولا يتجاوز أبدًا 2% و30%. كما تحدّ قيود محرك المخاطر الحلال أي مركز بـ20% من رأس المال.
- Source: GUARDRAILS.md rule 10; risk engine
- Enforced by: `quant_os_core.agents.profit_engine:dial_policy`, `quant_os_core.agents.trade_plan:PlanPolicy`, `quant_os_core.shariah.rules:halal_risk_limits`

### RK-04

- English: A daily loss of 3% or a drawdown of 15% halts new entries automatically.
- اردو: روزانہ 3% نقصان یا 15% گراوٹ پر نئی انٹریاں خود بخود رک جاتی ہیں۔
- العربية: خسارة يومية بنسبة 3% أو تراجع بنسبة 15% يوقف الدخول الجديد تلقائيًا.
- Source: GUARDRAILS.md rule 11; risk engine
- Enforced by: `quant_os_core.risk_engine.engine:RiskEngine.evaluate`, `quant_os_core.shariah.rules:halal_risk_limits`

### RK-05

- English: Protections lock new entries after a stop-loss streak, a run of losing trades or a drawdown across recent trades, and add a cooldown after each exit.
- اردو: حفاظتی تالے اسٹاپ لاس کے سلسلے، مسلسل نقصان یا حالیہ ٹریڈز میں گراوٹ کے بعد نئی انٹریاں روک دیتے ہیں، اور ہر اخراج کے بعد وقفہ رکھتے ہیں۔
- العربية: تقفل الحمايات الدخول الجديد بعد سلسلة من أوامر وقف الخسارة أو صفقات خاسرة متتالية أو تراجع عبر الصفقات الأخيرة، وتضيف فترة تهدئة بعد كل خروج.
- Source: GUARDRAILS.md rule 11; risk engine
- Enforced by: `quant_os_core.risk_engine.protections:default_protections`, `quant_os_core.risk_engine.protections:StoplossGuard`

### RK-06

- English: No pyramiding (no second entry in a symbol already held) and at most 3 new trades per day.
- اردو: پیرامڈنگ نہیں (پہلے سے موجود علامت میں دوسری انٹری نہیں) اور روزانہ زیادہ سے زیادہ 3 نئی ٹریڈز۔
- العربية: لا تعزيز للمراكز (لا دخول ثانٍ في رمز مملوك) وثلاث صفقات جديدة يوميًا كحد أقصى.
- Source: GUARDRAILS.md rule 12
- Enforced by: `quant_os_core.agents.trade_plan:PlanGovernor.check`, `quant_os_core.agents.trade_plan:PlanPolicy`

### RK-07

- English: Unknown state means no new entries: a broker error, an unreconciled position or stale market data stops entries until it clears.
- اردو: نامعلوم حالت میں نئی انٹری نہیں: بروکر کی خرابی، غیر مطابق پوزیشن یا پرانا مارکیٹ ڈیٹا ٹھیک ہونے تک انٹریاں روک دیتا ہے۔
- العربية: الحالة المجهولة تعني لا دخول جديد: خطأ الوسيط أو مركز غير مطابق أو بيانات سوق قديمة توقف الدخول حتى تُحل.
- Source: GUARDRAILS.md rule 13; risk engine
- Enforced by: `quant_os_core.ops.paper_loop:PaperTrader.step`, `quant_os_core.risk_engine.engine:RiskEngine.evaluate`

### RK-08

- English: The kill switch cancels pending orders and stops new entries; the risk engine halts every order while it is on.
- اردو: کِل سوئچ زیر التوا آرڈر منسوخ کرتا ہے اور نئی انٹریاں روکتا ہے؛ جب تک یہ آن ہے رسک انجن ہر آرڈر روکتا ہے۔
- العربية: مفتاح الإيقاف يلغي الأوامر المعلّقة ويوقف الدخول الجديد؛ ويوقف محرك المخاطر كل أمر ما دام مفعّلًا.
- Source: risk engine
- Enforced by: `quant_os_core.ops.paper_loop:PaperTrader.step`, `quant_os_core.risk_engine.engine:RiskEngine.evaluate`

### RK-09

- English: Edge over cost: a trade plan's expected edge must be at least 1.5 times its round-trip cost.
- اردو: لاگت پر برتری: ٹریڈ پلان کی متوقع برتری اس کی آنے جانے کی لاگت کا کم از کم 1.5 گنا ہونی چاہیے۔
- العربية: الميزة فوق التكلفة: يجب أن تكون الميزة المتوقعة لخطة الصفقة 1.5 ضعف تكلفة الذهاب والعودة على الأقل.
- Source: risk engine
- Enforced by: `quant_os_core.agents.profit_engine:check_edge`, `quant_os_core.agents.profit_engine:CostModel`

### RK-10

- English: Diversified core (cost guard): at most one rebalance a month, only when a fund is more than 2 points off target, never an order under $1; the Shariah gate checks every order.
- اردو: متنوع بنیادی پورٹ فولیو (لاگت گارڈ): مہینے میں زیادہ سے زیادہ ایک ری بیلنس، صرف جب کوئی فنڈ ہدف سے 2 پوائنٹس سے زیادہ ہٹا ہو، 1 ڈالر سے کم کا آرڈر کبھی نہیں؛ شرعی گیٹ ہر آرڈر جانچتا ہے۔
- العربية: المحفظة الأساسية المتنوعة (حارس التكلفة): إعادة توازن واحدة شهريًا كحد أقصى، فقط عندما يبتعد صندوق عن هدفه بأكثر من نقطتين، ولا أمر أقل من دولار واحد؛ وتفحص البوابة الشرعية كل أمر.
- Source: GUARDRAILS.md diversified core
- Enforced by: `quant_os_core.ops.core_loop:CoreRebalancer`

## AI and automation

### AI-01

- English: Analysts and language models can only veto or shrink a trade; they can never enlarge or open one.
- اردو: تجزیہ کار اور لینگویج ماڈلز صرف ٹریڈ رد یا چھوٹی کر سکتے ہیں؛ وہ کبھی اسے بڑا یا شروع نہیں کر سکتے۔
- العربية: يمكن للمحللين ونماذج اللغة رفض الصفقة أو تصغيرها فقط؛ ولا يمكنهم أبدًا تكبيرها أو فتحها.
- Source: GUARDRAILS.md rule 14
- Enforced by: `quant_os_core.agents.council:AnalystCouncil.deliberate`

## Operations

### OP-01

- English: Paper trading only: the broker adapter accepts only Alpaca's paper host.
- اردو: صرف پیپر ٹریڈنگ: بروکر اڈاپٹر صرف Alpaca کا پیپر ہوسٹ قبول کرتا ہے۔
- العربية: تداول تجريبي فقط: لا يقبل محوّل الوسيط إلا خادم Alpaca التجريبي.
- Source: GUARDRAILS.md rule 17
- Enforced by: `quant_os_core.ops.alpaca_paper:AlpacaPaperBroker.__init__`

### OP-02

- English: Live trading needs the two-key interlock: production mode plus a written risk acknowledgement.
- اردو: حقیقی ٹریڈنگ کے لیے دو چابیوں والا تالا ضروری ہے: پروڈکشن موڈ اور تحریری رسک اعتراف۔
- العربية: التداول الحقيقي يتطلب قفل المفتاحين: وضع الإنتاج مع إقرار مكتوب بالمخاطر.
- Source: GUARDRAILS.md rule 17
- Enforced by: `quant_os_core.settings:Settings.enforce_live_trading_interlock`

### OP-03

- English: Dashboard writes (kill switch, settings, journal, ledger) need the dashboard token; rules and locked settings cannot be changed from the dashboard.
- اردو: ڈیش بورڈ پر تبدیلیوں (کِل سوئچ، سیٹنگز، جرنل، لیجر) کے لیے ڈیش بورڈ ٹوکن ضروری ہے؛ قواعد اور مقفل سیٹنگز ڈیش بورڈ سے نہیں بدلی جا سکتیں۔
- العربية: تتطلب الكتابة من لوحة التحكم (مفتاح الإيقاف والإعدادات والسجل والدفتر) رمز اللوحة؛ ولا يمكن تغيير القواعد ولا الإعدادات المقفلة من اللوحة.
- Source: GUARDRAILS.md; risk engine
- Enforced by: `quant_os_core.ops.dashboard:make_handler`, `quant_os_core.ops.settings_store:SettingsStore.validate`
