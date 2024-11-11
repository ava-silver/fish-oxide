use std::collections::HashMap;
use std::sync::LazyLock;

use regex::Regex;

use crate::custom_rand::choice;
use crate::custom_rand::rand;

pub fn binomen() -> String {
    let data = vec![
        vec![
            "A", "AB", "AL", "AN", "AP", "AR", "AU", "BA", "BE", "BO", "BRA", "CA", "CAR", "CENT",
            "CHAE", "CHAN", "CHI", "CHRO", "CHRY", "CO", "CTE", "CY", "CYP", "DE", "E", "EU", "GA",
            "GAS", "GNA", "GO", "HE", "HIP", "HO", "HY", "LA", "LAB", "LE", "LI", "LO", "LU",
            "MAC", "ME", "MIC", "MO", "MU", "MY", "NA", "NAN", "NE", "NO", "O", "ON", "OP", "OS",
            "PA", "PER", "PHO", "PI", "PLA", "PLEU", "PO", "PSEU", "PTE", "RA", "RHI", "RHOM",
            "RU", "SAL", "SAR", "SCA", "SCOM", "SE", "SI", "STE", "TAU", "TEL", "THO", "TRI", "XE",
            "XI",
        ],
        vec![
            "BE", "BI", "BO", "BU", "CA", "CAM", "CAN", "CE", "CENT", "CHA", "CHEI", "CHI", "CHO",
            "CHY", "CI", "CIRR", "CO", "DI", "DO", "DON", "DOP", "GA", "GAS", "GO", "HI", "HYN",
            "LA", "LAB", "LE", "LEOT", "LI", "LICH", "LIS", "LO", "LOS", "LU", "LY", "MA", "ME",
            "MI", "MICH", "MO", "MU", "NA", "NE", "NEC", "NI", "NO", "NOCH", "NOP", "NOS", "PA",
            "PE", "PEN", "PHA", "PHI", "PHO", "PHY", "PHYO", "PI", "PIP", "PIS", "PO", "POG",
            "POPH", "RA", "RAE", "RAM", "REOCH", "RI", "RICH", "RIP", "RIS", "RO", "ROI", "ROP",
            "ROS", "RY", "RYN", "SE", "SO", "TA", "TE", "TEL", "THAL", "THE", "THO", "THOP", "THU",
            "TI", "TICH", "TO", "TOG", "TOP", "TOS", "VA", "XI", "XO",
        ],
        vec![
            "BIUS", "BUS", "CA", "CHUS", "CION", "CON", "CUS", "DA", "DES", "DEUS", "DON", "DUS",
            "GER", "GON", "GUS", "HUS", "LA", "LEA", "LIS", "LIUS", "LUS", "MA", "MIS", "MUS",
            "NA", "NIA", "NIO", "NIUS", "NOPS", "NUS", "PHEUS", "PHIS", "PIS", "PUS", "RA", "RAS",
            "RAX", "RIA", "RION", "RIS", "RUS", "RYS", "SA", "SER", "SIA", "SIS", "SUS", "TER",
            "TES", "TEUS", "THUS", "THYS", "TIA", "TIS", "TUS", "TYS",
        ],
        vec![
            "A", "AE", "AL", "AN", "AR", "AT", "AU", "AUST", "AY", "BA", "BAR", "BE", "BI", "BO",
            "CA", "CAL", "CAM", "CAN", "CAR", "CAU", "CE", "CHI", "CHRY", "COR", "CRY", "CU",
            "CYA", "DA", "DE", "DEN", "DI", "DIA", "DO", "DOR", "DU", "E", "FA", "FAS", "FES",
            "FI", "FLO", "FOR", "FRE", "FUR", "GLA", "GO", "HA", "HE", "HIP", "HO", "HYP", "I",
            "IM", "IN", "JA", "LA", "LAB", "LE", "LEU", "LI", "LO", "LU", "MA", "MAC", "MAR", "ME",
            "MO", "MOO", "MOR", "NA", "NE", "NI", "NIG", "NO", "O", "OR", "PA", "PAL", "PE", "PEC",
            "PHO", "PLA", "PLU", "PO", "PRO", "PU", "PUL", "RA", "RE", "RHOM", "RI", "RO", "ROST",
            "RU", "SA", "SAL", "SE", "SO", "SPI", "SPLEN", "STRIA", "TAU", "THO", "TRI", "TY", "U",
            "UN", "VA", "VI", "VIT", "VUL", "WAL", "XAN",
        ],
        vec![
            "BA", "BAR", "BER", "BI", "BO", "BOI", "BU", "CA", "CAN", "CAU", "CE", "CEL", "CHA",
            "CHEL", "CHOP", "CI", "CIA", "CIL", "CIO", "CO", "COS", "CU", "DA", "DE", "DEL", "DI",
            "DIA", "DO", "FAS", "FEL", "FI", "FOR", "GA", "GE", "GI", "HA", "HYN", "KE", "LA",
            "LAN", "LE", "LEA", "LEU", "LI", "LIA", "LO", "LON", "LOP", "MA", "ME", "MEN", "MI",
            "MIE", "MO", "NA", "NE", "NEA", "NEL", "NEN", "NI", "NIF", "NO", "NOI", "NOP", "NU",
            "PA", "PE", "PER", "PHA", "PHE", "PI", "PIN", "PO", "QUI", "RA", "RAC", "RE", "REN",
            "RES", "RI", "RIA", "RIEN", "RIF", "RO", "ROR", "ROS", "ROST", "RU", "RYTH", "SA",
            "SE", "SI", "SO", "SU", "TA", "TAE", "TE", "TER", "THAL", "THO", "THU", "TI", "TIG",
            "TO", "TU", "VA", "VE", "VES", "VI", "VIT", "XEL", "XI", "ZO",
        ],
        vec![
            "BEUS", "CA", "CENS", "CEPS", "CEUS", "CHA", "CHUS", "CI", "CUS", "DA", "DAX", "DENS",
            "DES", "DI", "DIS", "DUS", "FER", "GA", "GI", "GUS", "KEI", "KI", "LA", "LAS", "LI",
            "LIS", "LIUS", "LOR", "LUM", "LUS", "MA", "MIS", "MUS", "NA", "NEUS", "NI", "NII",
            "NIS", "NIUS", "NUS", "PIS", "PUS", "RA", "RE", "RI", "RIAE", "RIE", "RII", "RIO",
            "RIS", "RIX", "RONS", "RU", "RUM", "RUS", "SA", "SEUS", "SI", "SIS", "SUS", "TA",
            "TEUS", "THUS", "TI", "TIS", "TOR", "TUM", "TUS", "TZI", "ZI",
        ],
    ];
    let freq = vec![
        vec![
            27., 2., 4., 4., 2., 2., 2., 5., 2., 2., 3., 4., 2., 5., 3., 2., 2., 2., 3., 8., 3.,
            3., 2., 2., 7., 2., 3., 2., 2., 2., 6., 3., 2., 4., 5., 2., 5., 2., 3., 2., 2., 5., 5.,
            4., 2., 3., 2., 3., 2., 2., 9., 2., 2., 2., 7., 2., 2., 2., 2., 2., 5., 6., 2., 2., 2.,
            2., 2., 2., 2., 2., 2., 4., 2., 2., 2., 2., 3., 3., 2., 3.,
        ],
        vec![
            2., 2., 3., 3., 5., 2., 11., 6., 4., 2., 7., 2., 4., 4., 3., 3., 5., 4., 9., 2., 2.,
            4., 5., 13., 3., 3., 12., 3., 3., 2., 8., 3., 4., 15., 6., 2., 3., 10., 3., 3., 2., 2.,
            2., 8., 7., 3., 4., 20., 2., 2., 3., 4., 3., 2., 10., 2., 6., 2., 2., 5., 2., 2., 13.,
            2., 2., 14., 3., 2., 2., 9., 4., 2., 5., 42., 2., 4., 2., 6., 3., 3., 11., 2., 19., 2.,
            3., 2., 5., 3., 2., 4., 2., 27., 2., 2., 2., 2., 2., 2.,
        ],
        vec![
            3., 3., 7., 7., 3., 2., 3., 2., 5., 2., 13., 7., 2., 3., 4., 2., 13., 2., 2., 2., 24.,
            18., 13., 17., 12., 4., 2., 5., 3., 19., 3., 2., 2., 3., 7., 3., 2., 5., 2., 6., 29.,
            3., 2., 2., 2., 3., 4., 4., 16., 2., 6., 12., 5., 5., 6., 2.,
        ],
        vec![
            23., 3., 11., 6., 6., 3., 8., 2., 2., 2., 3., 3., 9., 3., 8., 2., 2., 3., 2., 2., 2.,
            2., 6., 2., 2., 3., 4., 3., 4., 2., 2., 2., 2., 2., 2., 15., 2., 4., 2., 2., 2., 2.,
            2., 2., 2., 2., 2., 5., 3., 2., 2., 3., 2., 2., 2., 7., 2., 2., 3., 3., 4., 4., 13.,
            7., 3., 10., 2., 2., 2., 5., 2., 3., 6., 4., 14., 2., 3., 2., 5., 2., 2., 3., 2., 3.,
            2., 2., 2., 3., 5., 2., 2., 3., 2., 3., 5., 2., 5., 2., 3., 3., 3., 3., 3., 7., 2., 3.,
            2., 4., 3., 2., 2., 2., 2.,
        ],
        vec![
            5., 2., 2., 4., 4., 2., 2., 10., 6., 2., 3., 5., 3., 2., 2., 6., 12., 2., 3., 6., 2.,
            22., 4., 4., 2., 7., 5., 6., 10., 2., 2., 2., 9., 7., 4., 2., 2., 2., 39., 3., 10., 3.,
            2., 20., 2., 10., 2., 2., 12., 9., 3., 8., 2., 4., 19., 5., 5., 3., 3., 12., 2., 9.,
            3., 2., 7., 3., 4., 3., 6., 2., 8., 5., 2., 4., 25., 2., 4., 3., 2., 26., 2., 2., 2.,
            21., 2., 2., 4., 6., 5., 3., 6., 4., 6., 2., 14., 2., 19., 2., 2., 2., 2., 21., 3.,
            14., 2., 3., 5., 2., 5., 2., 2., 2., 3.,
        ],
        vec![
            2., 7., 4., 3., 5., 2., 5., 2., 13., 6., 2., 2., 6., 8., 2., 4., 3., 4., 2., 5., 2.,
            5., 11., 3., 7., 19., 2., 2., 2., 11., 10., 4., 6., 12., 3., 15., 4., 6., 2., 18., 3.,
            3., 11., 4., 14., 2., 2., 3., 2., 13., 2., 3., 2., 4., 21., 7., 2., 10., 8., 13., 31.,
            2., 5., 5., 2., 2., 10., 68., 2., 3.,
        ],
    ];

    let mut name = choice(&data[0], Some(&freq[0])).to_string();
    let mut n = (rand() * 3.).trunc() as usize;
    for i in 0..n {
        name += &choice(&data[1], Some(&freq[1]));
    }
    name += &choice(&data[2], Some(&freq[2]));
    name += " ";
    name += &choice(&data[3], Some(&freq[3]));
    n = (rand() * 3.).trunc() as usize;
    for i in 0..n {
        name += &choice(&data[4], Some(&freq[4]));
    }
    name += &choice(&data[5], Some(&freq[5]));
    let re = Regex::new(r"([A-Z])\1\1+").unwrap();

    name = re.replace_all(&name, "$1$1").into_owned();
    let mut chars = name.chars();
    return format!(
        "{}{}",
        chars.next().unwrap(),
        chars.collect::<String>().to_lowercase()
    );
}

static HERSHEY_RAW: LazyLock<HashMap<i64, &'static str>> = LazyLock::new(|| {
    HashMap::from([
        (501, "  9I[RFJ[ RRFZ[ RMTWT"),
        (502, " 24G\\KFK[ RKFTFWGXHYJYLXNWOTP RKPTPWQXRYTYWXYWZT[K["),
        (503, " 19H]ZKYIWGUFQFOGMILKKNKSLVMXOZQ[U[WZYXZV"),
        (504, " 16G\\KFK[ RKFRFUGWIXKYNYSXVWXUZR[K["),
        (505, " 12H[LFL[ RLFYF RLPTP RL[Y["),
        (506, "  9HZLFL[ RLFYF RLPTP"),
        (507, " 23H]ZKYIWGUFQFOGMILKKNKSLVMXOZQ[U[WZYXZVZS RUSZS"),
        (508, "  9G]KFK[ RYFY[ RKPYP"),
        (509, "  3NVRFR["),
        (510, " 11JZVFVVUYTZR[P[NZMYLVLT"),
        (511, "  9G\\KFK[ RYFKT RPOY["),
        (512, "  6HYLFL[ RL[X["),
        (513, " 12F^JFJ[ RJFR[ RZFR[ RZFZ["),
        (514, "  9G]KFK[ RKFY[ RYFY["),
        (515, " 22G]PPUB FNGLIKKJNJSKVLXNZP[T[VZXXYVZSZNYKXIVGTFPF"),
        (516, " 14G\\KFK[ RKFTFWGXHYJYMXOWPTQKQ"),
        (
            517,
            " 25G]PPUB FNGLIKKJNJSKVLXNZP[T[VZXXYVZSZNYKXIVGTFPF RSWY]",
        ),
        (518, " 17G\\KFK[ RKFTFWGXHYJYLXNWOTPKP RRPY["),
        (519, " 21H\\YIWGTFPFMGKIKKLMMNOOUQWRXSYUYXWZT[P[MZKX"),
        (520, "  6JZRFR[ RKFYF"),
        (521, " 11G]KFKULXNZQ[S[VZXXYUYF"),
        (522, "  6I[JFR[ RZFR["),
        (523, " 12F^HFM[ RRFM[ RRFW[ R\\FW["),
        (524, "  6H\\KFY[ RYFK["),
        (525, "  7I[JFRPR[ RZFRP"),
        (526, "  9H\\YFK[ RKFYF RK[Y["),
        (601, " 18I\\XMX[ RXPVNTMQMONMPLSLUMXOZQ[T[VZXX"),
        (602, " 18H[LFL[ RLPNNPMSMUNWPXSXUWXUZS[P[NZLX"),
        (603, " 15I[XPVNTMQMONMPLSLUMXOZQ[T[VZXX"),
        (604, " 18I\\XFX[ RXPVNTMQMONMPLSLUMXOZQ[T[VZXX"),
        (605, " 18I[LSXSXQWOVNTMQMONMPLSLUMXOZQ[T[VZXX"),
        (606, "  9MYWFUFSGRJR[ ROMVM"),
        (607, " 23I\\XMX]W`VaTbQbOa RXPVNTMQMONMPLSLUMXOZQ[T[VZXX"),
        (608, " 11I\\MFM[ RMQPNRMUMWNXQX["),
        (609, "  9NVQFRGSFREQF RRMR["),
        (610, " 12MWRFSGTFSERF RSMS^RaPbNb"),
        (611, "  9IZMFM[ RWMMW RQSX["),
        (612, "  3NVRFR["),
        (613, " 19CaGMG[ RGQJNLMOMQNRQR[ RRQUNWMZM\\N]Q]["),
        (614, " 11I\\MMM[ RMQPNRMUMWNXQX["),
        (615, " 18I\\QMONMPLSLUMXOZQ[T[VZXXYUYSXPVNTMQM"),
        (616, " 18H[LMLb RLPNNPMSMUNWPXSXUWXUZS[P[NZLX"),
        (617, " 18I\\XMXb RXPVNTMQMONMPLSLUMXOZQ[T[VZXX"),
        (618, "  9KXOMO[ ROSPPRNTMWM"),
        (619, " 18J[XPWNTMQMNNMPNRPSUTWUXWXXWZT[Q[NZMX"),
        (620, "  9MYRFRWSZU[W[ ROMVM"),
        (621, " 11I\\MMMWNZP[S[UZXW RXMX["),
        (622, "  6JZLMR[ RXMR["),
        (623, " 12G]JMN[ RRMN[ RRMV[ RZMV["),
        (624, "  6J[MMX[ RXMM["),
        (625, " 10JZLMR[ RXMR[P_NaLbKb"),
        (626, "  9J[XMM[ RMMXM RM[X["),
        (710, "  6MWRYQZR[SZRY"),
    ])
});

// let hershey_cache = {};

pub fn compile_hershey(i: i64) -> (i64, i64, Vec<Vec<(f64, f64)>>) {
    // if (hershey_cache[i]){
    //   return hershey_cache[i];
    // }

    let Some(entry) = HERSHEY_RAW.get(&i) else {
        todo!();
    };
    let ord_r = 82;
    let mut bound = &mut entry[3..5].chars();
    let xmin = (bound.next().unwrap() as i64) - ord_r;
    let xmax = (bound.next().unwrap() as i64) - ord_r;
    let content = &entry[5..];
    let mut polylines = vec![vec![]];
    let mut j = 0;
    while (j < content.len()) {
        let digit = &content[j..j + 2];
        if (digit == " R") {
            polylines.push(vec![]);
        } else {
            let mut chars = digit.chars();
            let x = (chars.next().unwrap() as i64) - ord_r;
            let y = (chars.next().unwrap() as i64) - ord_r;
            polylines.last_mut().unwrap().push((x as f64, y as f64));
        }
        j += 2;
    }
    let data = (xmin, xmax, polylines);
    // hershey_cache[i] = data;
    return data;
}
