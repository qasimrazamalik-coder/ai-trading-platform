# ruff: noqa: E501, RUF001  (translated rule text keeps its own line lengths and Arabic punctuation)
"""Machine-readable rulebook: every hard rule the bot enforces, where it comes from and which
code enforces it.

Each ``enforced_by`` entry is ``module:qualified.name`` and is resolved by the tests, so a rule
cannot point at code that does not exist. The dashboard shows this list read-only; nothing in
the UI or the settings panel can change a rule. ``docs/RULEBOOK.md`` is generated from it by
:func:`markdown` (a test keeps the two in sync).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

SOURCES = ("GUARDRAILS.md", "IIFA", "AAOIFI", "risk engine")


@dataclass(frozen=True, slots=True)
class Rule:
    id: str
    group: str  # shariah | risk | ai | operations
    text: dict[str, str]  # en / ur / ar
    sources: tuple[str, ...]  # citation strings; each starts with one of SOURCES
    enforced_by: tuple[str, ...]  # "module:qualified.name"

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "group": self.group,
            "text": dict(self.text),
            "sources": list(self.sources),
            "enforced_by": [{"ref": ref, "file": file_of(ref)} for ref in self.enforced_by],
        }


_SH = "quant_os_core.shariah.rules"
_RISK = "quant_os_core.risk_engine.engine"


def _r(
    rule_id: str,
    group: str,
    en: str,
    ur: str,
    ar: str,
    sources: tuple[str, ...],
    enforced_by: tuple[str, ...],
) -> Rule:
    return Rule(rule_id, group, {"en": en, "ur": ur, "ar": ar}, sources, enforced_by)


RULES: tuple[Rule, ...] = (
    _r(
        "SH-01",
        "shariah",
        "Spot ownership only. Futures, forwards, options, binary options, CFDs, perpetuals, index products and prediction markets are blocked (R1, R2).",
        "صرف اسپاٹ ملکیت۔ فیوچرز، فارورڈز، آپشنز، بائنری آپشنز، سی ایف ڈی، پرپیچوئلز، انڈیکس پروڈکٹس اور پیشگوئی کی منڈیاں بند ہیں (R1، R2)۔",
        "التملّك الفوري فقط. العقود الآجلة والمستقبلية والخيارات والخيارات الثنائية وعقود الفروقات والعقود الدائمة ومنتجات المؤشرات وأسواق التوقعات محظورة (R1، R2).",
        ("GUARDRAILS.md rule 1", "IIFA Resolution 63"),
        (
            f"{_SH}:ShariahGate.check_instrument",
            f"{_SH}:BLOCKED_INSTRUMENT_TYPES",
            f"{_SH}:BLOCKED_ASSET_CLASSES",
        ),
    ),
    _r(
        "SH-02",
        "shariah",
        "No margin account and no buying above free cash: leverage is not permitted (R3).",
        "مارجن اکاؤنٹ نہیں اور دستیاب نقد سے زیادہ خریداری نہیں: لیوریج کی اجازت نہیں (R3)۔",
        "لا حساب هامش ولا شراء بأكثر من النقد المتاح: الرافعة المالية غير مسموحة (R3).",
        ("GUARDRAILS.md rules 2 and 8", "IIFA Resolution 63"),
        (f"{_SH}:ShariahGate.check_order",),
    ),
    _r(
        "SH-03",
        "shariah",
        "No interest, financing or swap charges: positions that carry them are blocked (R5).",
        "سود، فنانسنگ یا سواپ چارجز نہیں: ایسی پوزیشنیں بند ہیں (R5)۔",
        "لا فوائد ولا رسوم تمويل أو مبادلة: المراكز التي تحملها محظورة (R5).",
        ("GUARDRAILS.md rule 2",),
        (f"{_SH}:ShariahGate.check_order",),
    ),
    _r(
        "SH-04",
        "shariah",
        "No short selling: a sell larger than the quantity owned is refused (R4).",
        "شارٹ سیلنگ نہیں: ملکیت سے زیادہ فروخت رد کر دی جاتی ہے (R4)۔",
        "لا بيع على المكشوف: يُرفض أي بيع يتجاوز الكمية المملوكة (R4).",
        ("GUARDRAILS.md rule 3", "IIFA Resolution 63"),
        (
            f"{_SH}:ShariahGate.check_order",
            f"{_RISK}:RiskEngine.evaluate",
            f"{_SH}:halal_risk_limits",
        ),
    ),
    _r(
        "SH-05",
        "shariah",
        "Shares and funds must pass the AAOIFI ratios: interest-bearing debt and interest-bearing cash each under 30% of market value, non-compliant revenue under 5%, tangible assets at least one third (R6).",
        "شیئرز اور فنڈز کو AAOIFI کے تناسب پاس کرنا ہوں گے: سودی قرض اور سودی نقد ہر ایک مارکیٹ ویلیو کے 30% سے کم، غیر شرعی آمدنی 5% سے کم، ٹھوس اثاثے کم از کم ایک تہائی (R6)۔",
        "يجب أن تجتاز الأسهم والصناديق نسب أيوفي: الديون الربوية والنقد الربوي كلٌّ منهما أقل من 30% من القيمة السوقية، والإيرادات غير المتوافقة أقل من 5%، والأصول الملموسة ثلث على الأقل (R6).",
        ("GUARDRAILS.md rule 4", "AAOIFI Shariah Standard 21"),
        (f"{_SH}:ShariahGate.screen_company", f"{_SH}:ShariahProfile"),
    ),
    _r(
        "SH-06",
        "shariah",
        "Missing screening data blocks the buy: without a fund's or company's financials the bot cannot verify compliance (R6).",
        "اسکریننگ ڈیٹا نہ ہو تو خریداری بند: فنڈ یا کمپنی کے مالیاتی اعداد کے بغیر بوٹ شرعی مطابقت کی تصدیق نہیں کر سکتا (R6)۔",
        "غياب بيانات الفحص يمنع الشراء: دون البيانات المالية للصندوق أو الشركة لا يمكن التحقق من التوافق (R6).",
        ("GUARDRAILS.md rule 4", "AAOIFI Shariah Standard 21"),
        (f"{_SH}:ShariahGate.check_order",),
    ),
    _r(
        "SH-07",
        "shariah",
        "Sector screen: alcohol, pork, conventional finance and insurance, gambling, adult entertainment and tobacco are excluded (R7).",
        "شعبہ جاتی اسکرین: شراب، خنزیر، روایتی مالیات اور انشورنس، جوا، فحش تفریح اور تمباکو خارج ہیں (R7)۔",
        "فحص القطاع: الخمور ولحم الخنزير والتمويل والتأمين التقليديان والقمار والترفيه الإباحي والتبغ مستبعدة (R7).",
        ("GUARDRAILS.md rule 4", "AAOIFI Shariah Standard 21"),
        (f"{_SH}:ShariahGate.screen_company", f"{_SH}:CORE_EXCLUDED_SECTORS"),
    ),
    _r(
        "SH-08",
        "shariah",
        "Crypto only under a profile that enables it, and only tokens that pass the token screen; a missing token profile blocks the buy (R13).",
        "کرپٹو صرف اسی پروفائل میں جو اسے فعال کرے، اور صرف وہ ٹوکن جو ٹوکن اسکرین پاس کریں؛ ٹوکن پروفائل نہ ہو تو خریداری بند (R13)۔",
        "العملات المشفّرة فقط ضمن ملف يسمح بها، وفقط الرموز التي تجتاز فحص الرموز؛ وغياب ملف الرمز يمنع الشراء (R13).",
        ("GUARDRAILS.md rule 5",),
        (
            f"{_SH}:ShariahGate.screen_crypto",
            f"{_SH}:ShariahGate.check_instrument",
            f"{_SH}:ShariahGate.check_order",
        ),
    ),
    _r(
        "SH-09",
        "shariah",
        "Speculative forex is blocked (R10).",
        "سٹے بازی والا فاریکس بند ہے (R10)۔",
        "تداول العملات المضاربي محظور (R10).",
        ("GUARDRAILS.md rule 6", "AAOIFI Shariah Standard 1"),
        (f"{_SH}:ShariahGate.check_instrument",),
    ),
    _r(
        "SH-10",
        "shariah",
        "Venues not licensed in the owner's jurisdiction are blocked; Pakistani residents may not use offshore forex or CFD venues (R14).",
        "مالک کے دائرہ اختیار میں غیر لائسنس یافتہ پلیٹ فارم بند ہیں؛ پاکستانی رہائشی آف شور فاریکس یا سی ایف ڈی پلیٹ فارم استعمال نہیں کر سکتے (R14)۔",
        "المنصات غير المرخّصة في بلد المالك محظورة؛ ولا يجوز للمقيمين في باكستان استخدام منصات الفوركس أو عقود الفروقات الخارجية (R14).",
        ("GUARDRAILS.md rule 6",),
        (f"{_SH}:ShariahGate.check_order",),
    ),
    _r(
        "SH-11",
        "shariah",
        "Selling a position you own, to exit, is always allowed, even if the asset later fails the screen.",
        "اپنی ملکیتی پوزیشن بیچ کر نکلنا ہمیشہ جائز ہے، چاہے وہ اثاثہ بعد میں اسکرین میں ناکام ہو جائے۔",
        "بيع مركز مملوك للخروج مسموح دائمًا، حتى لو أخفق الأصل في الفحص لاحقًا.",
        ("GUARDRAILS.md rule 7",),
        (f"{_SH}:ShariahGate.check_order",),
    ),
    _r(
        "RK-01",
        "risk",
        "Leverage is fixed at 1.0; it cannot be raised by configuration.",
        "لیوریج 1.0 پر مقرر ہے؛ کنفیگریشن سے اسے بڑھایا نہیں جا سکتا۔",
        "الرافعة المالية ثابتة عند 1.0 ولا يمكن رفعها من الإعدادات.",
        ("GUARDRAILS.md rule 8", "risk engine"),
        (f"{_SH}:halal_risk_limits", f"{_RISK}:RiskEngine.evaluate"),
    ),
    _r(
        "RK-02",
        "risk",
        "Every entry has a stop below the entry, a target above it and a written invalidation condition before it is sent.",
        "ہر انٹری سے پہلے انٹری سے نیچے اسٹاپ، اوپر ہدف اور لکھی ہوئی منسوخی کی شرط لازمی ہے۔",
        "لكل دخول وقف أدنى من سعر الدخول وهدف أعلى منه وشرط إلغاء مكتوب قبل إرساله.",
        ("GUARDRAILS.md rule 9",),
        ("quant_os_core.agents.trade_plan:TradePlan.check_geometry",),
    ),
    _r(
        "RK-03",
        "risk",
        "Risk per trade at most 1% of equity and position at most 20% (balanced dial); 0.5% and 15% on conservative; the Growth dial is opt-in and never above 2% and 30%. The risk engine's halal limits also cap any position at 20% of equity.",
        "فی ٹریڈ خطرہ زیادہ سے زیادہ ایکویٹی کا 1% اور پوزیشن زیادہ سے زیادہ 20% (متوازن ڈائل)؛ محتاط پر 0.5% اور 15%؛ گروتھ ڈائل اختیاری ہے اور کبھی 2% اور 30% سے زیادہ نہیں۔ رسک انجن کی حلال حدیں بھی ہر پوزیشن کو ایکویٹی کے 20% تک محدود رکھتی ہیں۔",
        "المخاطرة لكل صفقة 1% من رأس المال كحد أقصى والمركز 20% كحد أقصى (الوضع المتوازن)؛ و0.5% و15% في الوضع المحافظ؛ ووضع النمو اختياري ولا يتجاوز أبدًا 2% و30%. كما تحدّ قيود محرك المخاطر الحلال أي مركز بـ20% من رأس المال.",
        ("GUARDRAILS.md rule 10", "risk engine"),
        (
            "quant_os_core.agents.profit_engine:dial_policy",
            "quant_os_core.agents.trade_plan:PlanPolicy",
            f"{_SH}:halal_risk_limits",
        ),
    ),
    _r(
        "RK-04",
        "risk",
        "A daily loss of 3% or a drawdown of 15% halts new entries automatically.",
        "روزانہ 3% نقصان یا 15% گراوٹ پر نئی انٹریاں خود بخود رک جاتی ہیں۔",
        "خسارة يومية بنسبة 3% أو تراجع بنسبة 15% يوقف الدخول الجديد تلقائيًا.",
        ("GUARDRAILS.md rule 11", "risk engine"),
        (f"{_RISK}:RiskEngine.evaluate", f"{_SH}:halal_risk_limits"),
    ),
    _r(
        "RK-05",
        "risk",
        "Protections lock new entries after a stop-loss streak, a run of losing trades or a drawdown across recent trades, and add a cooldown after each exit.",
        "حفاظتی تالے اسٹاپ لاس کے سلسلے، مسلسل نقصان یا حالیہ ٹریڈز میں گراوٹ کے بعد نئی انٹریاں روک دیتے ہیں، اور ہر اخراج کے بعد وقفہ رکھتے ہیں۔",
        "تقفل الحمايات الدخول الجديد بعد سلسلة من أوامر وقف الخسارة أو صفقات خاسرة متتالية أو تراجع عبر الصفقات الأخيرة، وتضيف فترة تهدئة بعد كل خروج.",
        ("GUARDRAILS.md rule 11", "risk engine"),
        (
            "quant_os_core.risk_engine.protections:default_protections",
            "quant_os_core.risk_engine.protections:StoplossGuard",
        ),
    ),
    _r(
        "RK-06",
        "risk",
        "No pyramiding (no second entry in a symbol already held) and at most 3 new trades per day.",
        "پیرامڈنگ نہیں (پہلے سے موجود علامت میں دوسری انٹری نہیں) اور روزانہ زیادہ سے زیادہ 3 نئی ٹریڈز۔",
        "لا تعزيز للمراكز (لا دخول ثانٍ في رمز مملوك) وثلاث صفقات جديدة يوميًا كحد أقصى.",
        ("GUARDRAILS.md rule 12",),
        (
            "quant_os_core.agents.trade_plan:PlanGovernor.check",
            "quant_os_core.agents.trade_plan:PlanPolicy",
        ),
    ),
    _r(
        "RK-07",
        "risk",
        "Unknown state means no new entries: a broker error, an unreconciled position or stale market data stops entries until it clears.",
        "نامعلوم حالت میں نئی انٹری نہیں: بروکر کی خرابی، غیر مطابق پوزیشن یا پرانا مارکیٹ ڈیٹا ٹھیک ہونے تک انٹریاں روک دیتا ہے۔",
        "الحالة المجهولة تعني لا دخول جديد: خطأ الوسيط أو مركز غير مطابق أو بيانات سوق قديمة توقف الدخول حتى تُحل.",
        ("GUARDRAILS.md rule 13", "risk engine"),
        (
            "quant_os_core.ops.paper_loop:PaperTrader.step",
            f"{_RISK}:RiskEngine.evaluate",
        ),
    ),
    _r(
        "RK-08",
        "risk",
        "The kill switch cancels pending orders and stops new entries; the risk engine halts every order while it is on.",
        "کِل سوئچ زیر التوا آرڈر منسوخ کرتا ہے اور نئی انٹریاں روکتا ہے؛ جب تک یہ آن ہے رسک انجن ہر آرڈر روکتا ہے۔",
        "مفتاح الإيقاف يلغي الأوامر المعلّقة ويوقف الدخول الجديد؛ ويوقف محرك المخاطر كل أمر ما دام مفعّلًا.",
        ("risk engine",),
        (
            "quant_os_core.ops.paper_loop:PaperTrader.step",
            f"{_RISK}:RiskEngine.evaluate",
        ),
    ),
    _r(
        "RK-09",
        "risk",
        "Edge over cost: a trade plan's expected edge must be at least 1.5 times its round-trip cost.",
        "لاگت پر برتری: ٹریڈ پلان کی متوقع برتری اس کی آنے جانے کی لاگت کا کم از کم 1.5 گنا ہونی چاہیے۔",
        "الميزة فوق التكلفة: يجب أن تكون الميزة المتوقعة لخطة الصفقة 1.5 ضعف تكلفة الذهاب والعودة على الأقل.",
        ("risk engine",),
        (
            "quant_os_core.agents.profit_engine:check_edge",
            "quant_os_core.agents.profit_engine:CostModel",
        ),
    ),
    _r(
        "RK-10",
        "risk",
        "Diversified core (cost guard): at most one rebalance a month, only when a fund is more than 2 points off target, never an order under $1; the Shariah gate checks every order.",
        "متنوع بنیادی پورٹ فولیو (لاگت گارڈ): مہینے میں زیادہ سے زیادہ ایک ری بیلنس، صرف جب کوئی فنڈ ہدف سے 2 پوائنٹس سے زیادہ ہٹا ہو، 1 ڈالر سے کم کا آرڈر کبھی نہیں؛ شرعی گیٹ ہر آرڈر جانچتا ہے۔",
        "المحفظة الأساسية المتنوعة (حارس التكلفة): إعادة توازن واحدة شهريًا كحد أقصى، فقط عندما يبتعد صندوق عن هدفه بأكثر من نقطتين، ولا أمر أقل من دولار واحد؛ وتفحص البوابة الشرعية كل أمر.",
        ("GUARDRAILS.md diversified core",),
        ("quant_os_core.ops.core_loop:CoreRebalancer",),
    ),
    _r(
        "AI-01",
        "ai",
        "Analysts and language models can only veto or shrink a trade; they can never enlarge or open one.",
        "تجزیہ کار اور لینگویج ماڈلز صرف ٹریڈ رد یا چھوٹی کر سکتے ہیں؛ وہ کبھی اسے بڑا یا شروع نہیں کر سکتے۔",
        "يمكن للمحللين ونماذج اللغة رفض الصفقة أو تصغيرها فقط؛ ولا يمكنهم أبدًا تكبيرها أو فتحها.",
        ("GUARDRAILS.md rule 14",),
        ("quant_os_core.agents.council:AnalystCouncil.deliberate",),
    ),
    _r(
        "OP-01",
        "operations",
        "Paper trading only: the broker adapter accepts only Alpaca's paper host.",
        "صرف پیپر ٹریڈنگ: بروکر اڈاپٹر صرف Alpaca کا پیپر ہوسٹ قبول کرتا ہے۔",
        "تداول تجريبي فقط: لا يقبل محوّل الوسيط إلا خادم Alpaca التجريبي.",
        ("GUARDRAILS.md rule 17",),
        ("quant_os_core.ops.alpaca_paper:AlpacaPaperBroker.__init__",),
    ),
    _r(
        "OP-02",
        "operations",
        "Live trading needs the two-key interlock: production mode plus a written risk acknowledgement.",
        "حقیقی ٹریڈنگ کے لیے دو چابیوں والا تالا ضروری ہے: پروڈکشن موڈ اور تحریری رسک اعتراف۔",
        "التداول الحقيقي يتطلب قفل المفتاحين: وضع الإنتاج مع إقرار مكتوب بالمخاطر.",
        ("GUARDRAILS.md rule 17",),
        ("quant_os_core.settings:Settings.enforce_live_trading_interlock",),
    ),
    _r(
        "OP-03",
        "operations",
        "Dashboard writes (kill switch, settings, journal, ledger) need the dashboard token; rules and locked settings cannot be changed from the dashboard.",
        "ڈیش بورڈ پر تبدیلیوں (کِل سوئچ، سیٹنگز، جرنل، لیجر) کے لیے ڈیش بورڈ ٹوکن ضروری ہے؛ قواعد اور مقفل سیٹنگز ڈیش بورڈ سے نہیں بدلی جا سکتیں۔",
        "تتطلب الكتابة من لوحة التحكم (مفتاح الإيقاف والإعدادات والسجل والدفتر) رمز اللوحة؛ ولا يمكن تغيير القواعد ولا الإعدادات المقفلة من اللوحة.",
        ("GUARDRAILS.md", "risk engine"),
        (
            "quant_os_core.ops.dashboard:make_handler",
            "quant_os_core.ops.settings_store:SettingsStore.validate",
        ),
    ),
)


def file_of(ref: str) -> str:
    """``"quant_os_core.shariah.rules:X"`` -> ``"packages/core/src/quant_os_core/shariah/rules.py"``."""

    module = ref.split(":", 1)[0]
    return "packages/core/src/" + module.replace(".", "/") + ".py"


def resolve(ref: str) -> object:
    """Import the object an ``enforced_by`` reference names; raises if it does not exist."""

    module_name, _, qualname = ref.partition(":")
    if not qualname:
        raise ValueError(f"reference needs 'module:name': {ref}")
    target: object = importlib.import_module(module_name)
    for part in qualname.split("."):
        target = getattr(target, part)
    return target


def as_json(lang: str = "en") -> dict[str, Any]:
    """What ``GET /api/rulebook`` returns. Read-only by construction: there is no setter."""

    from quant_os_core.ops import i18n

    code = i18n.normalize(lang)
    return {
        "lang": code,
        "editable": False,
        "rules": [{**rule.as_dict(), "label": rule.text[code]} for rule in RULES],
    }


def markdown() -> str:
    """The text of ``docs/RULEBOOK.md``."""

    lines = [
        "# Rulebook",
        "",
        "Every hard rule the bot enforces, with its source and the code that enforces it.",
        "Generated from `packages/core/src/quant_os_core/ops/rulebook.py` (`rulebook.markdown()`);",
        "a test fails when this file and the code drift apart. The dashboard shows the same list",
        "read-only. Changing a rule needs a code change, a test, a reason in `CHANGELOG.md` and,",
        "for Shariah rules, a scholar's written sign-off (see `GUARDRAILS.md`).",
        "",
    ]
    titles = {
        "shariah": "Shariah",
        "risk": "Risk",
        "ai": "AI and automation",
        "operations": "Operations",
    }
    for group, title in titles.items():
        lines += [f"## {title}", ""]
        for rule in (r for r in RULES if r.group == group):
            lines += [
                f"### {rule.id}",
                "",
                f"- English: {rule.text['en']}",
                f"- اردو: {rule.text['ur']}",
                f"- العربية: {rule.text['ar']}",
                f"- Source: {'; '.join(rule.sources)}",
                "- Enforced by: " + ", ".join(f"`{ref}`" for ref in rule.enforced_by),
                "",
            ]
    return "\n".join(lines)
