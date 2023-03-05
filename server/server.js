// components
const http = require('http');
// const _kurulum = require('./kurulum');

const _python = require('./worker/py-denetcisi');

const server = http.createServer((request, response) => {
        // kullanici daha az yetkisi olan "denetci"ye donusturuluyor
        // if (process.setuid && process.getuid) {
        //     process.setgid(_kurulum.denetciKimligi);
        //     process.setuid(_kurulum.denetciKimligi);
        // }
    
        // response.setHeader('Access-Control-Allow-Origin', _kurulum.applicationserver);
        response.setHeader('Access-Control-Allow-Origin', '*');
        response.setHeader('Access-Control-Allow-Headers', '*');
        response.setHeader('Vary', 'Origin');

        // if (request.socket.remoteAddress == _kurulum.uygulamaSunucusuKimligi || request.method === 'POST') {
        if (request.method == 'POST') {
            console.log('----method:'+request.method);
            let gelen = '';
            request.on('data', (parca) => {
                gelen += parca;
            });
            request.on('end', () => {
                // duzenleyici icerigi belgeye yaziliyor
                gelen = JSON.parse(gelen);
                // console.log('gelen:');
                // console.log(gelen);
                if(gelen.process != 'NULL'){
                    _python.giden(gelen, response);
                }
                else{
                    _python.giden('NULL', response);
                }
                console.log('----------------------------------')
            });
        }
        else {
            console.log('----method:'+request.method);
            let gelen = '';
            request.on('data', (parca) => {
                gelen += parca;
            });
            console.log('gelen:'+gelen);

            let cikti = { basari: 1 };
            cikti = JSON.stringify(cikti);
            response.writeHead(200, { 'Content-Type': 'application/json' });
            response.end(cikti);
        }
});

// server starting
server.listen(5500, () => {
    console.log('Listening 5500 port.');
});
// server.timeout = 25000;