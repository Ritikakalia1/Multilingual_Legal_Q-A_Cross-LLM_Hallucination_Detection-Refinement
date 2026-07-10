# test_model.py — Training-distribution sanity check
#
# Goal: verify each adapter (en/hi/pa/ne) performs well on questions we KNOW
# are in the training distribution (verbatim/near-verbatim from ipc_qa.json),
# and separately confirm it still struggles/hallucinates on BNS (which we
# already know is absent from the dataset). This isolates "adapter is broken"
# from "adapter was never taught this."

from model_loader import ModelLoader

loader = ModelLoader()
loader.load_base_model()

# ── In-distribution test set ──
# These are the same 3 IPC Q&As (title/extent, operation extent, certain laws)
# pulled directly from your dataset for each language, with ground-truth
# answers included so we can eyeball-compare instead of just reading output
# in isolation.

IN_DISTRIBUTION = {
    "en": [
        {
            "question": "What is the title and extent of operation of the Indian Penal Code?",
            "expected": "The title is 'The Indian Penal Code' and its operation extends to the "
                        "punishment of offences committed within India, and beyond but which by "
                        "law may be tried within India. It also includes extension of the Code to "
                        "extra-territorial offences.",
        },
        {
            "question": "Where does the operation of 'The Indian Penal Code' extend to, and does "
                         "it include extra-territorial offences?",
            "expected": "The operation of 'The Indian Penal Code' extends to the punishment of "
                        "offences committed within India, and beyond but which by law may be tried "
                        "within India. It also includes extension of the Code to extra-territorial "
                        "offences.",
        },
        {
            "question": "What does the Indian Penal Code stipulate about certain laws?",
            "expected": "The Indian Penal Code specifies that certain laws are not to be affected "
                        "by this Act.",
        },
    ],
    "hi": [
        {
            "question": "भारतीय पेनल कोड के संचालन का शीर्षक और विस्तार क्या है?",
            "expected": "शीर्षक है 'भारतीय पेनल कोड' और इसका ऑपरेशन भारत के भीतर किए अपराधों के "
                        "दंड तक पहुँचता है, और इसके अलावा जो भी कानून के द्वारा जाँच की जा सकती है। "
                        "इसमें अतिरिक्त-क्षेत्रीय अपराधों का विस्तार भी शामिल है।",
        },
        {
            "question": "'भारतीय पेनल कोड' की प्रक्रिया कहाँ होती है, और क्या यह अतिरिक्त गुनाहों को "
                         "शामिल करता है?",
            "expected": "'भारतीय पेनल कोड' भारत के भीतर किए अपराधों के दंड को विस्तार करता है, और "
                        "इसके अलावा जो भारत के भीतर कोशिश की जा सकती है। इसमें कोड का विस्तार भी "
                        "अतिरिक्त-क्षेत्रीय अपराधों के लिए सम्मिलित है।",
        },
        {
            "question": "भारतीय पेनल कोड कुछ नियमों के बारे में क्या निर्धारित करता है?",
            "expected": "भारतीय पेनल कोड यह निर्धारित करता है कि इस कार्य से कुछ नियम प्रभावित "
                        "नहीं होते।",
        },
    ],
    "pa": [
        {
            "question": "ਇੰਡੀਅਨ ਪੀਨਲ ਕੋਡ ਦੀ ਕਾਰਵਾਈ ਦਾ ਸਿਰਲੇਖ ਅਤੇ ਹੱਦ ਕੀ ਹੈ?",
            "expected": "ਸਿਰਲੇਖ 'ਭਾਰਤੀ ਦੰਡ ਸੰਹਿਤਾ' ਹੈ ਅਤੇ ਇਸਦੀ ਕਾਰਵਾਈ ਭਾਰਤ ਦੇ ਅੰਦਰ ਕੀਤੇ ਗਏ ਅਪਰਾਧਾਂ ਦੀ "
                        "ਸਜ਼ਾ ਤੱਕ ਵਿਸਤ੍ਰਿਤ ਹੈ, ਅਤੇ ਇਸ ਤੋਂ ਬਾਹਰ, ਪਰ ਜਿਨ੍ਹਾਂ 'ਤੇ ਕਾਨੂੰਨ ਦੁਆਰਾ ਭਾਰਤ ਦੇ ਅੰਦਰ "
                        "ਮੁਕੱਦਮਾ ਚਲਾਇਆ ਜਾ ਸਕਦਾ ਹੈ। ਇਸ ਵਿੱਚ ਸੰਹਿਤਾ ਨੂੰ ਵਾਧੂ ਖੇਤਰੀ ਅਪਰਾਧਾਂ ਤੱਕ ਵਧਾਉਣਾ ਵੀ "
                        "ਸ਼ਾਮਲ ਹੈ।",
        },
        {
            "question": "'ਇੰਡੀਅਨ ਪੀਨਲ ਕੋਡ' ਦੀ ਕਾਰਵਾਈ ਕਿੱਥੇ ਤੱਕ ਫੈਲੀ ਹੋਈ ਹੈ, ਅਤੇ ਕੀ ਇਸ ਵਿੱਚ ਖੇਤਰ ਤੋਂ "
                         "ਬਾਹਰਲੇ ਅਪਰਾਧ ਸ਼ਾਮਲ ਹਨ?",
            "expected": "'ਦ ਇੰਡੀਅਨ ਪੀਨਲ ਕੋਡ' ਦਾ ਸੰਚਾਲਨ ਭਾਰਤ ਦੇ ਅੰਦਰ ਕੀਤੇ ਗਏ ਅਪਰਾਧਾਂ ਦੀ ਸਜ਼ਾ ਤੱਕ "
                        "ਵਿਸਤ੍ਰਿਤ ਹੈ, ਅਤੇ ਇਸ ਤੋਂ ਬਾਹਰ, ਪਰ ਭਾਰਤ ਦੇ ਅੰਦਰ ਕਾਨੂੰਨ ਦੁਆਰਾ ਮੁਕੱਦਮਾ ਚਲਾਇਆ ਜਾ "
                        "ਸਕਦਾ ਹੈ। ਇਸ ਵਿੱਚ ਸੰਹਿਤਾ ਨੂੰ ਵਾਧੂ ਖੇਤਰੀ ਅਪਰਾਧਾਂ ਤੱਕ ਵਧਾਉਣਾ ਵੀ ਸ਼ਾਮਲ ਹੈ।",
        },
        {
            "question": "ਇੰਡੀਅਨ ਪੀਨਲ ਕੋਡ ਕੁਝ ਕਾਨੂੰਨਾਂ ਬਾਰੇ ਕੀ ਨਿਰਧਾਰਤ ਕਰਦਾ ਹੈ?",
            "expected": "ਇੰਡੀਅਨ ਪੀਨਲ ਕੋਡ ਸਪੱਸ਼ਟ ਕਰਦਾ ਹੈ ਕਿ ਕੁਝ ਕਾਨੂੰਨ ਇਸ ਐਕਟ ਦੁਆਰਾ ਪ੍ਰਭਾਵਿਤ ਨਹੀਂ ਹੋਣਗੇ।",
        },
    ],
    "ne": [
        {
            "question": "भारतीय दण्ड संहिताको कार्यको शीर्षक र सीमा के हो?",
            "expected": "शीर्षक 'भारतीय दण्ड संहिता' हो र यसको सञ्चालन भारत भित्र भएका अपराधहरूको "
                        "दण्डसम्म फैलिएको छ, र त्यसभन्दा बाहिर तर भारत भित्र कानूनद्वारा मुद्दा "
                        "चलाउन सकिन्छ। यसले अतिरिक्त-क्षेत्रीय अपराधहरूमा संहिताको विस्तार पनि "
                        "समावेश गर्दछ।",
        },
        {
            "question": "'भारतीय दण्ड संहिता' को सञ्चालन कहाँसम्म फैलिएको छ, र के यसले अतिरिक्त "
                         "क्षेत्रीय अपराधहरू समावेश गर्दछ?",
            "expected": "'भारतीय दण्ड संहिता' को सञ्चालनले भारत भित्र र त्यसभन्दा बाहिरका अपराधहरूको "
                        "सजायसम्म विस्तार गर्दछ तर जसलाई कानूनद्वारा भारतभित्र मुद्दा चलाउन सकिन्छ। "
                        "यसले अतिरिक्त-क्षेत्रीय अपराधहरूमा संहिताको विस्तार पनि समावेश गर्दछ।",
        },
        {
            "question": "भारतीय दण्ड संहिताले कतिपय कानुनहरूको बारेमा के व्यवस्था गर्छ?",
            "expected": "भारतीय दण्ड संहिताले निर्दिष्ट गरेको छ कि केहि कानूनहरू यस ऐनबाट प्रभावित "
                        "हुँदैनन्।",
        },
    ],
}

# ── Out-of-distribution control ──
# Same BNS-style question in each language. We EXPECT these to be weak/wrong
# since BNS isn't in the training data — this just confirms the failure mode
# is consistent and isolated to BNS, not spilling into IPC/CrPC/Constitution.
OOD_CONTROL = {
    "en": "What section of the BNS corresponds to Section 302 IPC?",
    "hi": "IPC की धारा 302 के अनुरूप BNS की कौन सी धारा है?",
    "pa": "IPC ਦੀ ਧਾਰਾ 302 ਦੇ ਬਰਾਬਰ BNS ਦੀ ਕਿਹੜੀ ਧਾਰਾ ਹੈ?",
    "ne": "IPC को धारा ३०२ सँग मिल्ने BNS को धारा कुन हो?",
}


def run_in_distribution_tests():
    print("\n" + "=" * 70)
    print("IN-DISTRIBUTION TESTS (known training data — should be accurate)")
    print("=" * 70)

    for lang, cases in IN_DISTRIBUTION.items():
        print(f"\n########## LANGUAGE: {lang} ##########")
        for i, case in enumerate(cases, 1):
            print(f"\n--- {lang} Q{i} ---")
            print("Question:", case["question"])
            answer = loader.generate(lang=lang, question=case["question"])
            print("Model answer:   ", answer)
            print("Expected answer:", case["expected"])


def run_ood_control():
    print("\n" + "=" * 70)
    print("OUT-OF-DISTRIBUTION CONTROL (BNS — expected to be weak/hallucinated)")
    print("=" * 70)

    for lang, question in OOD_CONTROL.items():
        print(f"\n--- {lang} (BNS control) ---")
        print("Question:", question)
        answer = loader.generate(lang=lang, question=question)
        print("Model answer:", answer)


if __name__ == "__main__":
    run_in_distribution_tests()
    run_ood_control()

    print("\n" + "=" * 70)
    print("ADAPTER STATE CHECK")
    print("=" * 70)
    print("Attached adapters:", list(loader.base_model.peft_config.keys()))
    print("Active adapter:", loader.base_model.active_adapter)