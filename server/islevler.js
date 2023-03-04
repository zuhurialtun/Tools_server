// gomulu bilesenler
const fs = require('fs-extra');

// kurulum
const _kurulum = require('./kurulum');

//
//
//
//
//
// ciktiyi gonder
//
//
//
//
//
const gonderici = (response, giden) => {
    // giden = giden.split('\n');
    giden = JSON.stringify(giden);
    console.log(giden)
    response.writeHead(200, { 'Content-Type': 'application/json' });
    response.end(giden);
};

//
//
//
//
//
// ust dizin bulucu
//
//
//
//
//
const ustDizinBulucu = (kullaniciKI) => {
    // 1000 1000 dizin olusturuluyor
    return parseInt(kullaniciKI / 1000);
};

//
//
//
//
//
// belge adi bulucu
//
//
//
//
//
const belgeAdiBulucu = (gelen) => {
    // java
    if (gelen.duzenleyici.dil == 'java') {
        let icerik = gelen.duzenleyici.icerik.split('\n');
        let bulundu = false;
        for (let i = 0; i < icerik.length; i++) {
            if (icerik[i].includes('class') && bulundu == false) {
                var sinifAdi = icerik[i].split('class')[1];
                sinifAdi = sinifAdi.trim().split(' ')[0];
                sinifAdi = sinifAdi.trim();
                // ilki aliniyor
                bulundu = true;
            }
        }
        var belgeAdi = `${sinifAdi}.${_kurulum.dil[gelen.duzenleyici.dil].uzanti}`;
    }
    // diger diller
    else {
        var belgeAdi = `${gelen.kullanici.kullanici_ki}.${_kurulum.dil[gelen.duzenleyici.dil].uzanti}`;
    }
    return {
        belgeAdi: belgeAdi,
        sinifAdi: sinifAdi,
    };
};

//
//
//
//
//
// uygulama sunucusundan geleni dosyaya yaziyor
//
//
//
//
//
const yazici = (gelen) => {
    //
    // girdiler dile ozgu kucuk degisikliklerden dolayi on islemden geciyor
    //
    let onIslemlisoruGirdisi = '';

    // javascript
    if (gelen.duzenleyici.dil == 'javascript') {
        onIslemlisoruGirdisi = 'girdi=' + gelen.soru.girdi + ';';
    }

    // php
    else if (gelen.duzenleyici.dil == 'php') {
        // gelen icerigin basinda <?php yoksa eklenecek
        // temel bir denetim yapiliyor, her duruma bakilmiyor, gerek de yok
        let zorunluBaslangic = '<?php';
        var gelenIcerikBaslangici = gelen.duzenleyici.icerik.slice(0, 5);
        if (gelenIcerikBaslangici != zorunluBaslangic) {
            gelen.duzenleyici.icerik = zorunluBaslangic.concat('\n').concat(gelen.duzenleyici.icerik);
        }
        // tek tirnaktan kurtariliyor
        if (gelen.soru.girdi) gelen.soru.girdi = gelen.soru.girdi.replace(/'/g, "\\'");
        // basina sonuna tek tirnak ekleniyor ki degisken olarak yorumlansin $s='abc'; gibi.
        onIslemlisoruGirdisi = "$girdi='" + gelen.soru.girdi + "';";
    }

    // python
    else if (gelen.duzenleyici.dil == 'python') {
        onIslemlisoruGirdisi = 'girdi=' + gelen.soru.girdi;
    }

    //
    // soru girdisiyle ilgili seyler ekleniyor
    //
    if (gelen.soru != false) {
        // dillere göre soru girdisi taslaginin konumu bulunuyor
        let soruGirdisiKonumu = '';

        // javascript
        if (gelen.duzenleyici.dil == 'javascript') {
            soruGirdisiKonumu = _kurulum.soruGirdisiKonumu + _kurulum.dil[gelen.duzenleyici.dil].uzanti + '-girdisi.js';
        }

        // php
        else if (gelen.duzenleyici.dil == 'php') {
            soruGirdisiKonumu = _kurulum.soruGirdisiKonumu + _kurulum.dil[gelen.duzenleyici.dil].uzanti + '-girdisi.php';
        }

        // python
        else if (gelen.duzenleyici.dil == 'python') {
            soruGirdisiKonumu = _kurulum.soruGirdisiKonumu + _kurulum.dil[gelen.duzenleyici.dil].uzanti + '-girdisi.py';
        } else {
        }

        // tum diller
        let soruGirdisiCiktisi = fs.readFileSync(`${soruGirdisiKonumu}`, 'utf8');
        gelen.duzenleyici.icerik += '\n\n';
        gelen.duzenleyici.icerik += onIslemlisoruGirdisi;
        gelen.duzenleyici.icerik += '\n\n';
        soruGirdisiCiktisi = soruGirdisiCiktisi.replace(/___islev___/g, gelen.soru.islev_adi);
        gelen.duzenleyici.icerik += soruGirdisiCiktisi;
    }

    // belge konumu bulunuyor
    let ustDizin = ustDizinBulucu(gelen.kullanici.kullanici_ki);
    let belgeKonumu = `${_kurulum.belgeKonumu}${ustDizin}/${gelen.kullanici.kullanici_ki}/`;

    // belge adi bulunuyor
    belgeAdi = belgeAdiBulucu(gelen);

    // yaziliyor
    fs.outputFileSync(`${belgeKonumu}${belgeAdi['belgeAdi']}`, gelen.duzenleyici.icerik, (hata) => {});
    // return object
    return {
        dizin: belgeKonumu,
        belge: belgeAdi.belgeAdi,
        sinif: belgeAdi.sinifAdi,
    };
};

module.exports = { gonderici, yazici, ustDizinBulucu, belgeAdiBulucu };
