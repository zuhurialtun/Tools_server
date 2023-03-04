// bilesenler
// const _kurulum = require('../kurulum');
const { performance } = require('perf_hooks');
const { execFile } = require('child_process');
const { gonderici } = require('../islevler');

module.exports = {
    giden: function (belge, response) {
        let calismaBaslangici = performance.now();
        let giden = {};
        
        execFile('python', ['./worker/modelworker.py',belge], (yurutmeHatasi, dilCiktisi, dilHatasi) => {
            // basarim
            let calismaSuresi = Math.round(performance.now() - calismaBaslangici);
            let bellekKullanimi = Math.round(process.memoryUsage().heapUsed / 1000);

            dilCiktisi = dilCiktisi.split('\n');
            dilCiktisi = (dilCiktisi[4]);

            if (yurutmeHatasi) giden['yurutmeHatasi'] = yurutmeHatasi.code;
            if (dilCiktisi) giden['dilCiktisi'] = dilCiktisi;
            giden['calismaSuresi'] = calismaSuresi;
            giden['bellekKullanimi'] = bellekKullanimi;

            // console.log(giden);
            // uygulama sunucusuna gonderim
            gonderici(response, giden);
        });
    },
};
