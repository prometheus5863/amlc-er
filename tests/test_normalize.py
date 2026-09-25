from amlc.normalize import norm_address, norm_name, skeleton


def core(n):
    return norm_name(n)[1]


def test_noise_prefixes_and_legal_forms():
    assert core("-- Holloway Peak Inc Seafood") == "holloway peak seafood"
    assert core("TORRES INC. INTEGRATED") == core("Torres Integrated Inc.") == "torres integrated"
    assert core("S 6 First Spac L.L.C.") == core("S 6 First Spac LLC")
    assert norm_name("Nagavi S0il Pvt. Ltd.")[2] == "pvt_ltd"


def test_digit_letter_swaps_but_real_numbers_kept():
    assert core("Nagavi S0il Pvt. Ltd.") == "nagavi soil"
    assert core("Shri Specia1ist (India) India Limited") == core("Mr Specialist (India) India (Limited)")
    assert core("84 Congo Industrie EURL") == "84 congo industrie"


def test_domains_and_handles():
    n = norm_name("wilfordhancock.com")
    assert n[1] == "wilfordhancock" and n[3]
    assert norm_name("@Yeagerstaking")[3]


def test_alias():
    assert norm_name("Onyxumbra F/K/A Oncology Care Inc")[5] == "oncology care"


def test_french_legal_forms():
    assert norm_name("Ape Tech  S.A.S")[2] == "sas"
    assert core("SASU HMB Residence Participations") == "hmb residence participations"


def test_indic_scripts():
    assert norm_name("राम मार्केटिंग प्राइवेट लिमिटेड")[1:3] == ("ram marketing", "pvt_ltd")
    assert norm_name("শিবা সিস্টেমস প্রাইভেট লিমিটেড")[4]  # Bengali is transliterated too
    assert skeleton("intaranesanala")[:4] == skeleton("international")[:4]


def test_addresses():
    a = norm_address("#1303 Cattle Trl, Austin, Texas", "US")
    assert a[0] == "1303 cattle trail, austin, texas" and a[1] == "1303"
    assert norm_address("0015703 BYBEE DR, PORTLAND, OR", "US")[1] == "15703"
    assert norm_address("OAK GROVE ROAD, N/A, ELKIN, NC", "US")[0] == "oak grove road, elkin, nc"
    fr = norm_address("(67) R. SAINTE-THÉRÈSE, ROUBAIX, Nord", "France")
    assert fr[0].startswith("67 rue sainte therese")
    assert norm_address("Plot No-1154, Bhubaneswar, Orissa 751019", "India")[4] == "751019"
