#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

# src/templates/__init__.py
# Arabic
from .ara.verb_gender_plurality_agreement_imperative import (
    AraVerbGenderPluralityAgreementImperative,
)
from .ara.verb_gender_plurality_agreement_indicative import (
    AraVerbGenderPluralityAgreementIndicative,
)
from .ara.adjective_gender_plurality_agreement import (
    AraAdjectiveGenderPluralityAgreement,
)
from .ara.number_gender_plurality_agreement import AraNumberGenderPluralityAgreement


# Finnish
from .fin.verb_plurality_agreement_indicative import (
    FinnVerbPluralityAgreementIndicative,
)

from .fin.verb_plurality_agreement_imperative import (
    FinnVerbPluralityAgreementImperative,
)

from .fin.adjective_case_plurality_agreement import FinAdjectiveCasePluralityAgreement

from .fin.vowel_harmony import FinVowelHarmony

from .fin.lexical_casing_city import FinLexicalCasingCity


# Russian
from .rus.verb_gender_plurality_agreement_indicative import (
    RusVerbGenderPluralityAgreementIndicative,
)

from .rus.verb_gender_plurality_agreement_imperative import (
    RusVerbGenderPluralityAgreementImperative,
)

from .rus.adjective_case_plurality_gender_agreement import (
    RusAdjectiveCasePluralityAgreement,
)

from .rus.motion_verbs import RusMotionVerbs


# Turkish
from .tur.verb_plurality_agreement_indicative import TurVerbPluralityAgreementIndicative
from .tur.verb_plurality_agreement_imperative import TurVerbPluralityAgreementImperative
from .tur.verb_plurality_agreement_evidentiality import (
    TurVerbPluralityAgreementEvidentiality,
)
from .tur.vowel_harmony import TurVowelHarmony
from .tur.adjective_plurality_agreement import TurAdjectivePluralityAgreement

# Hebrew
from .heb.adjective_gender_plurality_agreement import (
    HebAdjectiveGenderPluralityAgreement,
)
from .heb.verb_gender_plurality_agreement_indicative import (
    HebVerbGenderPluralityAgreementIndicative,
)
from .heb.verb_gender_plurality_agreement_imperative import (
    HebVerbGenderPluralityAgreementImperative,
)
from .heb.number_gender_plurality_agreement import HebNumberGenderPluralityAgreement

# English
from .eng.adjective_plurality_agreement import EngAdjectivePluralityAgreement
from .eng.verb_plurality_agreement_imperative import EngVerbPluralityAgreementImperative
from .eng.verb_plurality_agreement_indicative import EngVerbPluralityAgreementIndicative

# Dummy reference to ensure imports are considered used
_ = AraVerbGenderPluralityAgreementImperative
_ = AraVerbGenderPluralityAgreementIndicative
_ = AraAdjectiveGenderPluralityAgreement
_ = AraNumberGenderPluralityAgreement

_ = FinnVerbPluralityAgreementIndicative
_ = FinnVerbPluralityAgreementImperative
_ = FinAdjectiveCasePluralityAgreement
_ = FinVowelHarmony
_ = FinLexicalCasingCity

_ = RusVerbGenderPluralityAgreementIndicative
_ = RusVerbGenderPluralityAgreementImperative
_ = RusAdjectiveCasePluralityAgreement
_ = RusMotionVerbs

_ = TurVerbPluralityAgreementIndicative
_ = TurVerbPluralityAgreementImperative
_ = TurVerbPluralityAgreementEvidentiality
_ = TurVowelHarmony
_ = TurAdjectivePluralityAgreement

_ = HebAdjectiveGenderPluralityAgreement
_ = HebVerbGenderPluralityAgreementIndicative
_ = HebVerbGenderPluralityAgreementImperative
_ = HebNumberGenderPluralityAgreement

_ = EngAdjectivePluralityAgreement
_ = EngVerbPluralityAgreementImperative
_ = EngVerbPluralityAgreementIndicative
