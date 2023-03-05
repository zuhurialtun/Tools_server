// bilesenler
// const _kurulum = require('../kurulum');
const { performance } = require('perf_hooks');
const { execFile } = require('child_process');
const { gonderici } = require('../islevler');

module.exports = {
    giden: function (belge, response) {
        let calismaBaslangici = performance.now();
        let giden = {};
        
        execFile('python', ['./server/worker/jiroTools.py',belge.randomID,belge.process,belge.file_list], (yurutmeHatasi, dilCiktisi, dilHatasi) => {
            // basarim
            let calismaSuresi = Math.round(performance.now() - calismaBaslangici);
            let bellekKullanimi = Math.round(process.memoryUsage().heapUsed / 1000);

            // console.log('---Test');
            // console.log(dilCiktisi);
            // console.log('---Test');

            if (yurutmeHatasi) giden['yurutmeHatasi'] = yurutmeHatasi.code;
            if (dilCiktisi) giden['dilCiktisi'] = dilCiktisi;
            giden['calismaSuresi'] = calismaSuresi;
            giden['bellekKullanimi'] = bellekKullanimi;
            // giden['dilHatasi'] = dilHatasi;

            // console.log(giden);
            // uygulama sunucusuna gonderim
            gonderici(response, giden);
        });
    },
};
