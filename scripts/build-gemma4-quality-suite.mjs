#!/usr/bin/env node
/**
 * Build data/gemma4-quality-suite.json (v2) from corpus + fixed task templates.
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');

const EN_CORPUS_SNIPPETS = {
  mortgage: 'When taking a mortgage, the lender sets an individual margin on top of the reference rate and requires amortization when loan-to-value exceeds regulatory thresholds. Borrowers should compare the annual percentage rate including fees, not only the nominal interest rate, and keep an emergency fund for rate shocks.',
  legal: 'Consumer credit laws require clear pre-contract disclosure of the annual percentage rate, total cost, and withdrawal rights before the agreement becomes binding. Remote contracts often include a cooling-off period during which the consumer may cancel without penalty.',
  medical: 'Hypertension management combines lifestyle measures such as reduced sodium intake, regular exercise, and weight control with home blood pressure monitoring. Clinicians may prescribe medication when readings remain above guideline thresholds despite lifestyle changes.',
};

function doc(corpus, lang, topic, idx = 0) {
  if (lang === 'en') {
    return { text: EN_CORPUS_SNIPPETS[topic], language: 'en', topic };
  }
  const matches = corpus.documents.filter((d) => d.language === lang && d.topic === topic);
  return matches[idx % matches.length];
}

const WRITING_EXTRA = {
  en: {
    mortgage: {
      prompt: 'Write about 100 words in English explaining amortization requirements when loan-to-value exceeds 70%, stress testing, and why borrowers should keep an emergency fund alongside mortgage payments.',
      keywords: ['amort', 'loan', 'stress', 'payment', 'mortgage', 'fund', 'rate', 'bank', 'borrow', 'value'],
    },
    legal: {
      prompt: 'Write about 100 words in English on employment contract basics: written terms, notice periods, and protection against unfair dismissal under typical labor law frameworks.',
      keywords: ['employ', 'contract', 'notice', 'worker', 'dismiss', 'law', 'right', 'term', 'employer', 'protection'],
    },
    medical: {
      prompt: 'Write about 100 words in English on type 2 diabetes self-management: blood glucose monitoring, diet, exercise, and when to seek urgent care.',
      keywords: ['diabetes', 'glucose', 'blood', 'diet', 'exercise', 'insulin', 'doctor', 'sugar', 'health', 'care'],
    },
  },
  sv: {
    mortgage: {
      prompt: 'Skriv cirka 100 ord på svenska om pantbrev, lagfart och tillträdesdag vid bostadsköp. Förklara varför banken kräver säkerheter innan utbetalning.',
      keywords: ['pantbrev', 'lagfart', 'tillträde', 'bank', 'köp', 'bostad', 'säkerhet', 'kostnad', 'fastighet', 'lån'],
    },
    legal: {
      prompt: 'Skriv cirka 100 ord på svenska om hyresrätt och besittningsskydd: när hyresvärd får säga upp och vilken myndighet som prövar tvister.',
      keywords: ['hyra', 'hyresgäst', 'uppsägning', 'besittning', 'saklig', 'hyresnämnd', 'avtal', 'bostad', 'rätt', 'tvist'],
    },
    medical: {
      prompt: 'Skriv cirka 100 ord på svenska om diabetes typ 2 och egenvård: blodsockermätning, kost, motion och när man ska söka akut vård.',
      keywords: ['diabetes', 'blodsocker', 'kost', 'motion', 'vård', 'insulin', 'läkare', 'hälsa', 'patient', 'behandling'],
    },
  },
  tr: {
    mortgage: {
      prompt: 'Türkçe yaklaşık 100 kelime: konut kredisinde erken ödeme, sigorta ve ipotek masraflarını açıklayın. Bankanın ödeme öncesi hangi belgeleri istediğini belirtin.',
      keywords: ['kredi', 'ipotek', 'sigorta', 'ödeme', 'banka', 'masraf', 'konut', 'erken', 'belge', 'faiz'],
    },
    legal: {
      prompt: 'Türkçe yaklaşık 100 kelime: KVKK kapsamında kişisel verilerin korunması, aydınlatma yükümlülüğü ve veri sahibi başvuru hakları.',
      keywords: ['kvkk', 'veri', 'kişisel', 'aydınlatma', 'hak', 'sorumlu', 'kanun', 'başvuru', 'koruma', 'işleme'],
    },
    medical: {
      prompt: 'Türkçe yaklaşık 100 kelime: tip 2 diyabet yönetimi — kan şekeri takibi, beslenme, egzersiz ve acil başvuru belirtileri.',
      keywords: ['diyabet', 'şeker', 'kan', 'beslenme', 'egzersiz', 'insülin', 'doktor', 'sağlık', 'hasta', 'tedavi'],
    },
  },
};

function buildJsonTasks() {
  const templates = [
    {
      id: 'json-en-mortgage',
      language: 'en',
      topic: 'mortgage',
      prompt: (t) => `Read the text and extract JSON only (no markdown):\n{"borrower_name":string,"loan_amount_usd":number,"interest_rate_percent":number,"loan_term_years":number,"property_city":string,"ltv_percent":number,"rate_type":"fixed"|"variable"}\n\nTEXT:\n${t}\n\nJSON:`,
      text: 'Loan application summary — Borrower: Maria Chen. Property in Austin, Texas. Purchase price USD 480,000 with loan amount USD 384,000 (80% LTV). Approved 30-year variable-rate mortgage at 6.25% APR. Closing 14 May 2026.',
      gold: { borrower_name: 'Maria Chen', loan_amount_usd: 384000, interest_rate_percent: 6.25, loan_term_years: 30, property_city: 'Austin', ltv_percent: 80, rate_type: 'variable' },
      fields: ['borrower_name', 'loan_amount_usd', 'interest_rate_percent', 'loan_term_years', 'property_city', 'ltv_percent', 'rate_type'],
    },
    {
      id: 'json-en-legal',
      language: 'en',
      topic: 'legal',
      prompt: (t) => `Extract JSON only:\n{"law_name":string,"notice_days":number,"party":string,"penalty_usd":number,"binding":boolean}\n\nTEXT:\n${t}\n\nJSON:`,
      text: 'Consumer Credit Act — Lender must give 14-day cooling-off notice before binding. Borrower: James Okonkwo. Early repayment penalty USD 0 if within notice window. Agreement becomes binding after notice expires.',
      gold: { law_name: 'Consumer Credit Act', notice_days: 14, party: 'James Okonkwo', penalty_usd: 0, binding: true },
      fields: ['law_name', 'notice_days', 'party', 'penalty_usd', 'binding'],
    },
    {
      id: 'json-en-medical',
      language: 'en',
      topic: 'medical',
      prompt: (t) => `Extract JSON only:\n{"patient_age":number,"systolic_bp":number,"diastolic_bp":number,"diagnosis":string,"medication_mg":number,"follow_up_weeks":number}\n\nTEXT:\n${t}\n\nJSON:`,
      text: 'Patient summary — Male age 58, BP 148/92 mmHg. Primary hypertension diagnosed. Started lisinopril 10 mg daily. Follow-up in 2 weeks. Advised salt restriction and walking.',
      gold: { patient_age: 58, systolic_bp: 148, diastolic_bp: 92, diagnosis: 'hypertension', medication_mg: 10, follow_up_weeks: 2 },
      fields: ['patient_age', 'systolic_bp', 'diastolic_bp', 'diagnosis', 'medication_mg', 'follow_up_weeks'],
    },
    {
      id: 'json-sv-mortgage',
      language: 'sv',
      topic: 'mortgage',
      prompt: (t) => `Extrahera JSON endast:\n{"bank":string,"loan_amount_sek":number,"ltv_percent":number,"rate_type":string,"amort_percent":number}\n\nTEXT:\n${t}\n\nJSON:`,
      text: 'Erbjudande från Nordea: bolån 3 200 000 kronor till villa i Uppsala. Belåningsgrad 65 procent. Rörlig ränta 4,89 procent. Amorteringskrav 1 procent per år enligt FI.',
      gold: { bank: 'Nordea', loan_amount_sek: 3200000, ltv_percent: 65, rate_type: 'rörlig', amort_percent: 1 },
      fields: ['bank', 'loan_amount_sek', 'ltv_percent', 'rate_type', 'amort_percent'],
    },
    {
      id: 'json-sv-legal',
      language: 'sv',
      topic: 'legal',
      prompt: (t) => `Extrahera JSON:\n{"law_reference":string,"withdrawal_days":number,"contract_type":string,"consumer_country":string,"penalty_applies":boolean}\n\nTEXT:\n${t}\n\nJSON:`,
      text: 'Konsumentkreditlagen (2010:1846) reglerar kreditavtal i Sverige. Vid distansavtal om konsumentkrediter har konsumenten 14 dagars ångerrätt. Avtalet gällde privatlån 150 000 kronor. Ingen vite vid ångring inom fristen.',
      gold: { law_reference: '2010:1846', withdrawal_days: 14, contract_type: 'konsumentkredit', consumer_country: 'Sverige', penalty_applies: false },
      fields: ['law_reference', 'withdrawal_days', 'contract_type', 'consumer_country', 'penalty_applies'],
    },
    {
      id: 'json-sv-medical',
      language: 'sv',
      topic: 'medical',
      prompt: (t) => `Extrahera JSON:\n{"patient_age":number,"systolic_bp":number,"diastolic_bp":number,"diagnosis":string,"medication_mg":number,"follow_up_weeks":number}\n\nTEXT:\n${t}\n\nJSON:`,
      text: 'Patient 62 år, blodtryck 152/94. Hypertoni grad 1. Metoprolol 50 mg dagligen insatt. Återbesök om 3 veckor. Rekommenderad DASH-kost och promenader.',
      gold: { patient_age: 62, systolic_bp: 152, diastolic_bp: 94, diagnosis: 'hypertoni', medication_mg: 50, follow_up_weeks: 3 },
      fields: ['patient_age', 'systolic_bp', 'diastolic_bp', 'diagnosis', 'medication_mg', 'follow_up_weeks'],
    },
    {
      id: 'json-tr-mortgage',
      language: 'tr',
      topic: 'mortgage',
      prompt: (t) => `JSON çıkarın:\n{"bank":string,"loan_amount_try":number,"term_years":number,"rate_percent":number,"down_payment_percent":number}\n\nMETİN:\n${t}\n\nJSON:`,
      text: 'Ziraat Bankası onayı: 2 400 000 TL konut kredisi, 15 yıl vade, yıllık %38,5 faiz. Peşinat oranı %25. DASK ve konut sigortası zorunlu.',
      gold: { bank: 'Ziraat', loan_amount_try: 2400000, term_years: 15, rate_percent: 38.5, down_payment_percent: 25 },
      fields: ['bank', 'loan_amount_try', 'term_years', 'rate_percent', 'down_payment_percent'],
    },
    {
      id: 'json-tr-legal',
      language: 'tr',
      topic: 'legal',
      prompt: (t) => `JSON çıkarın:\n{"law_number":string,"withdrawal_days":number,"topic":string,"authority":string,"penalty_applies":boolean}\n\nMETİN:\n${t}\n\nJSON:`,
      text: '6502 sayılı Tüketicinin Korunması Hakkında Kanun mesafeli sözleşmelerde bilgilendirme zorunludur. Finansal hizmetlerde 14 gün cayma hakkı tanınabilir. Uyuşmazlıklar Tüketici Hakem Heyetine gider. Cayma süresinde ceza uygulanmaz.',
      gold: { law_number: '6502', withdrawal_days: 14, topic: 'tüketici kredisi', authority: 'Tüketici Hakem Heyeti', penalty_applies: false },
      fields: ['law_number', 'withdrawal_days', 'topic', 'authority', 'penalty_applies'],
    },
    {
      id: 'json-tr-medical',
      language: 'tr',
      topic: 'medical',
      prompt: (t) => `JSON çıkarın:\n{"patient_age":number,"systolic_bp":number,"diastolic_bp":number,"diagnosis":string,"medication_mg":number,"follow_up_weeks":number}\n\nMETİN:\n${t}\n\nJSON:`,
      text: 'Hasta özeti — 58 yaşında erkek, tansiyon 148/92 mmHg. Primer hipertansiyon. Lisinopril 10 mg günde bir. İki hafta sonra kontrol. Tuz kısıtlaması önerildi.',
      gold: { patient_age: 58, systolic_bp: 148, diastolic_bp: 92, diagnosis: 'hipertansiyon', medication_mg: 10, follow_up_weeks: 2 },
      fields: ['patient_age', 'systolic_bp', 'diastolic_bp', 'diagnosis', 'medication_mg', 'follow_up_weeks'],
    },
  ];

  return templates.map((t) => ({
    id: t.id,
    language: t.language,
    topic: t.topic,
    prompt: t.prompt(t.text),
    gold: t.gold,
    fields: t.fields,
  }));
}

const MCQ_EXTRA = [
  { id: 'mcq-en-mortgage-payment', language: 'en', topic: 'mortgage', prompt: 'A borrower pays $2,400/month on a 30-year fixed mortgage. Which factor does NOT directly change the scheduled principal portion in the first years?\nA) Interest rate\nB) Loan balance\nC) Borrower age\nD) Remaining term\nAnswer with one letter.', correct: 'C' },
  { id: 'mcq-sv-mortgage-belaning', language: 'sv', topic: 'mortgage', prompt: 'Om bostadens värde sjunker men skulden är oförändrad, vad händer med belåningsgraden?\nA) Den minskar\nB) Den ökar\nC) Oförändrad\nD) Avgiftsfri\nSvara med en bokstav.', correct: 'B' },
  { id: 'mcq-tr-mortgage-oran', language: 'tr', topic: 'mortgage', prompt: 'Konut kredisinde peşinat %20 ise kredi/değer oranı yaklaşık nedir?\nA) %20\nB) %60\nC) %80\nD) %100\nTek harf yazın.', correct: 'C' },
  { id: 'mcq-en-legal-cooling', language: 'en', topic: 'legal', prompt: 'A remote consumer credit agreement often allows withdrawal without penalty within how many days in many EU-style regimes?\nA) 3 days\nB) 7 days\nC) 14 days\nD) 60 days\nOne letter only.', correct: 'C' },
  { id: 'mcq-sv-legal-hyra', language: 'sv', topic: 'legal', prompt: 'Vilken myndighet prövar ofta hyrestvister i Sverige?\nA) Finansinspektionen\nB) Hyresnämnden\nC) Skatteverket\nD) Kronofogden\nSvara med en bokstav.', correct: 'B' },
  { id: 'mcq-tr-legal-kvkk', language: 'tr', topic: 'legal', prompt: 'Türkiye\'de kişisel verilerin korunması hangi kanunla düzenlenir?\nA) 6098 sayılı TBK\nB) 6698 sayılı KVKK\nC) 5510 sayılı SGK\nD) 6502 sayılı TKHK\nTek harf.', correct: 'B' },
  { id: 'mcq-en-medical-bp', language: 'en', topic: 'medical', prompt: 'Normal adult blood pressure is generally considered below which reading?\nA) 140/90 mmHg\nB) 120/80 mmHg\nC) 160/100 mmHg\nD) 100/60 mmHg\nOne letter.', correct: 'B' },
  { id: 'mcq-sv-medical-diabetes', language: 'sv', topic: 'medical', prompt: 'Vilket är ett typiskt egenvårdsmål vid diabetes typ 2?\nA) Öka sockerintag\nB) Regelbunden fysisk aktivitet\nC) Sluta mäta blodsocker\nD) Undvika läkarbesök\nSvara med en bokstav.', correct: 'B' },
  { id: 'mcq-tr-medical-tuz', language: 'tr', topic: 'medical', prompt: 'Hipertansiyon tedavisinde genelde önerilen yaşam tarzı değişikliği hangisidir?\nA) Tuz alımını artırmak\nB) Tuz alımını azaltmak\nC) Egzersizden kaçınmak\nD) Sıvı alımını durdurmak\nTek harf.', correct: 'B' },
  { id: 'mcq-en-mortgage-escrow', language: 'en', topic: 'mortgage', prompt: 'An escrow account in a US-style mortgage typically holds funds for:\nA) Groceries and utilities only\nB) Property taxes and insurance\nC) Stock investments\nD) Credit card payments\nOne letter.', correct: 'B' },
  { id: 'mcq-sv-legal-las', language: 'sv', topic: 'legal', prompt: 'Vilken lag reglerar grundläggande anställningsskydd i Sverige?\nA) Hyreslagen\nB) LAS\nC) Ärvdabalken\nD) LOU\nSvara med en bokstav.', correct: 'B' },
  { id: 'mcq-tr-mortgage-sigorta', language: 'tr', topic: 'mortgage', prompt: 'Türkiye\'de konut kredisiyle birlikte genelde hangi sigorta zorunludur?\nA) Seyahat sigortası\nB) DASK / deprem sigortası\nC) Hayat sigortası hariç hiçbiri\nD) Evcil hayvan sigortası\nTek harf.', correct: 'B' },
];

const RC_EXTRA = [
  { id: 'rc-en-mortgage', language: 'en', topic: 'mortgage', prompt: 'Passage: The borrower put down 20% on a $400,000 home, financing the rest with a 30-year loan.\nQuestion: What was the loan amount in dollars? Reply with digits only.', gold_answers: ['320000', '320,000'] },
  { id: 'rc-sv-legal', language: 'sv', topic: 'legal', prompt: 'Text: Konsumenten har 14 dagars ångerrätt vid distansavtal om konsumentkrediter.\nFråga: Hur många dagars ångerrätt? Svara med en siffra.', gold_answers: ['14'] },
  { id: 'rc-tr-mortgage', language: 'tr', topic: 'mortgage', prompt: 'Metin: Peşinat oranı %25 olan konut kredisinde kalan finansman oranı %75\'tir.\nSoru: Finansman oranı yüzde kaçtır? Yalnızca sayı yazın.', gold_answers: ['75'] },
  { id: 'rc-en-medical', language: 'en', topic: 'medical', prompt: 'Passage: The patient was started on metformin 500 mg twice daily and referred for HbA1c recheck in 3 months.\nQuestion: How many months until HbA1c recheck? Number only.', gold_answers: ['3', 'three'] },
  { id: 'rc-sv-medical', language: 'sv', topic: 'medical', prompt: 'Text: Patienten fick metoprolol 50 mg dagligen och återbesök om 3 veckor.\nFråga: Hur många veckor till återbesök? Svara med siffra.', gold_answers: ['3', 'tre'] },
  { id: 'rc-tr-legal', language: 'tr', topic: 'legal', prompt: 'Metin: Mesafeli sözleşmelerde cayma hakkı 14 gün olarak uygulanabilir.\nSoru: Cayma süresi kaç gün? Yalnızca sayı.', gold_answers: ['14', 'on dört'] },
];

const IF_EXTRA = [
  { id: 'if-en-legal-numbered', language: 'en', topic: 'legal', prompt: 'Reply with exactly two numbered lines (1. and 2.) stating two consumer rights before signing a credit contract. No other text.', rules: { type: 'numbered_count', count: 2, prefix: '1.' } },
  { id: 'if-sv-mortgage-bullets', language: 'sv', topic: 'mortgage', prompt: 'Lista exakt tre punkter (varje rad börjar med \"- \") om fördelar med att amortera bolån. Ingen annan text.', rules: { type: 'bullet_count', count: 3, prefix: '- ' } },
  { id: 'if-tr-legal-yesno', language: 'tr', topic: 'legal', prompt: 'Soru: Mesafeli sözleşmelerde bilgilendirme formu verilmesi zorunlu mudur? Yalnızca \"Evet\" veya \"Hayır\" yazın.', rules: { type: 'exact_one_of', options: ['Evet', 'Hayır', 'evet', 'hayır'] } },
  { id: 'if-en-medical-sentences', language: 'en', topic: 'medical', prompt: 'Answer in exactly two complete sentences about why home blood pressure monitoring helps hypertension care. No bullet list.', rules: { type: 'sentence_count', count: 2 } },
  { id: 'if-sv-medical-yesno', language: 'sv', topic: 'medical', prompt: 'Fråga: Bör patienter med hypertoni minska saltintaget? Svara endast \"Ja\" eller \"Nej\".', rules: { type: 'exact_one_of', options: ['Ja', 'Nej', 'ja', 'nej'] } },
  { id: 'if-tr-mortgage-bullets', language: 'tr', topic: 'mortgage', prompt: 'Konut kredisinde dikkat edilmesi gereken tam üç maddeyi \"- \" ile başlayan satırlar halinde yazın. Başka metin eklemeyin.', rules: { type: 'bullet_count', count: 3, prefix: '- ' } },
];

async function main() {
  const corpus = JSON.parse(await fs.readFile(path.join(root, 'data', 'benchmark-corpus.json'), 'utf8'));
  const v1 = JSON.parse(await fs.readFile(path.join(root, 'data', 'gemma4-quality-suite.json'), 'utf8'));

  const langs = ['en', 'sv', 'tr'];
  const topics = ['mortgage', 'legal', 'medical'];

  const writingTasks = [...v1.categories.domain_writing.tasks];
  for (const lang of langs) {
    for (const topic of topics) {
      const extra = WRITING_EXTRA[lang][topic];
      writingTasks.push({
        id: `write-${lang}-${topic}-b`,
        language: lang,
        topic,
        prompt: extra.prompt,
        keywords: extra.keywords,
      });
    }
  }

  const summarizeTasks = [];
  for (const lang of langs) {
    for (const topic of topics) {
      const d = doc(corpus, lang, topic, 1);
      const labels = { en: 'Summarize in exactly two sentences', sv: 'Sammanfatta i exakt två meningar', tr: 'Tam iki cümleyle özetleyin' };
      summarizeTasks.push({
        id: `sum-${lang}-${topic}`,
        language: lang,
        topic,
        prompt: `${labels[lang]}:\n\n${d.text}`,
        required_keywords: d.text.split(/\s+/).filter((w) => w.length > 6).slice(0, 6).map((w) => w.replace(/[^\p{L}\p{N}]/gu, '').toLowerCase()).filter(Boolean),
        sentence_count: 2,
      });
    }
  }

  const classifyTasks = [];
  const classifySamples = [
    { lang: 'en', topic: 'mortgage', text: 'The lender required a 20% down payment and set a 30-year amortization schedule with escrow for taxes.' },
    { lang: 'en', topic: 'legal', text: 'The consumer may withdraw from the remote credit agreement within fourteen days without stating a reason.' },
    { lang: 'sv', topic: 'medical', text: 'Patienten rekommenderades att mäta blodtryck dagligen och minska saltintaget vid hypertoni.' },
    { lang: 'tr', topic: 'mortgage', text: 'Banka %25 peşinat sonrası 15 yıllık konut kredisi onayladı ve DASK poliçesi istedi.' },
    { lang: 'sv', topic: 'legal', text: 'Hyresgästen har besittningsskydd och hyresvärd måste ha saklig grund vid uppsägning.' },
    { lang: 'tr', topic: 'medical', text: 'Hipertansiyon tanısıyla lisinopril 10 mg reçete edildi ve tuz kısıtlaması önerildi.' },
  ];
  for (const s of classifySamples) {
    const labelPrompt = { en: 'Reply with ONE word only: mortgage, legal, or medical.', sv: 'Svara med ETT ord: mortgage, legal eller medical.', tr: 'Yalnızca BİR kelime: mortgage, legal veya medical.' };
    classifyTasks.push({
      id: `cls-${s.lang}-${s.topic}`,
      language: s.lang,
      topic: s.topic,
      prompt: `${labelPrompt[s.lang]}\n\nText:\n${s.text}`,
      expected: s.topic,
    });
  }

  const suite = {
    version: 2,
    description: 'Gemma 4 quality eval v2 — expanded EN/SV/TR tasks across 7 categories.',
    categories: {
      domain_writing: {
        weight: 0.18,
        min_words: 60,
        target_words: 100,
        max_words: 220,
        tasks: writingTasks,
      },
      json_extraction: {
        weight: 0.18,
        tasks: buildJsonTasks(),
      },
      mcq: {
        weight: 0.16,
        tasks: [...v1.categories.mcq.tasks, ...MCQ_EXTRA],
      },
      reading_comprehension: {
        weight: 0.12,
        tasks: [...v1.categories.reading_comprehension.tasks, ...RC_EXTRA],
      },
      instruction_following: {
        weight: 0.12,
        tasks: [...v1.categories.instruction_following.tasks, ...IF_EXTRA],
      },
      summarization: {
        weight: 0.12,
        tasks: summarizeTasks,
      },
      classification: {
        weight: 0.12,
        tasks: classifyTasks,
      },
    },
  };

  const out = path.join(root, 'data', 'gemma4-quality-suite.json');
  await fs.writeFile(out, JSON.stringify(suite, null, 2));
  const counts = Object.fromEntries(Object.entries(suite.categories).map(([k, v]) => [k, v.tasks.length]));
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  console.log('Wrote', out, counts, 'total', total);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
