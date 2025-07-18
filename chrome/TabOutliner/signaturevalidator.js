/** @preserve Copyright 2012, 2013, 2014, 2015 by Vladyslav Volovyk. All Rights Reserved. */

function _base64ToArrayBuffer(base64) {
    var binary_string =  atob(base64);
    var len = binary_string.length;
    var bytes = new Uint8Array( len );
    for (var i = 0; i < len; i++)        {
        bytes[i] = binary_string.charCodeAt(i);
    }
    return bytes;
}


function strToUTF8Arr (sDOMStr) {

  var aBytes, nChr, nStrLen = sDOMStr.length, nArrLen = 0;

  /* mapping... */

  for (var nMapIdx = 0; nMapIdx < nStrLen; nMapIdx++) {
    nChr = sDOMStr.charCodeAt(nMapIdx);
    nArrLen += nChr < 0x80 ? 1 : nChr < 0x800 ? 2 : nChr < 0x10000 ? 3 : nChr < 0x200000 ? 4 : nChr < 0x4000000 ? 5 : 6;
  }

  aBytes = new Uint8Array(nArrLen);

  /* transcription... */

  for (var nIdx = 0, nChrIdx = 0; nIdx < nArrLen; nChrIdx++) {
    nChr = sDOMStr.charCodeAt(nChrIdx);
    if (nChr < 128) {
      /* one byte */
      aBytes[nIdx++] = nChr;
    } else if (nChr < 0x800) {
      /* two bytes */
      aBytes[nIdx++] = 192 + (nChr >>> 6);
      aBytes[nIdx++] = 128 + (nChr & 63);
    } else if (nChr < 0x10000) {
      /* three bytes */
      aBytes[nIdx++] = 224 + (nChr >>> 12);
      aBytes[nIdx++] = 128 + (nChr >>> 6 & 63);
      aBytes[nIdx++] = 128 + (nChr & 63);
    } else if (nChr < 0x200000) {
      /* four bytes */
      aBytes[nIdx++] = 240 + (nChr >>> 18);
      aBytes[nIdx++] = 128 + (nChr >>> 12 & 63);
      aBytes[nIdx++] = 128 + (nChr >>> 6 & 63);
      aBytes[nIdx++] = 128 + (nChr & 63);
    } else if (nChr < 0x4000000) {
      /* five bytes */
      aBytes[nIdx++] = 248 + (nChr >>> 24);
      aBytes[nIdx++] = 128 + (nChr >>> 18 & 63);
      aBytes[nIdx++] = 128 + (nChr >>> 12 & 63);
      aBytes[nIdx++] = 128 + (nChr >>> 6 & 63);
      aBytes[nIdx++] = 128 + (nChr & 63);
    } else /* if (nChr <= 0x7fffffff) */ {
      /* six bytes */
      aBytes[nIdx++] = 252 + (nChr >>> 30);
      aBytes[nIdx++] = 128 + (nChr >>> 24 & 63);
      aBytes[nIdx++] = 128 + (nChr >>> 18 & 63);
      aBytes[nIdx++] = 128 + (nChr >>> 12 & 63);
      aBytes[nIdx++] = 128 + (nChr >>> 6 & 63);
      aBytes[nIdx++] = 128 + (nChr & 63);
    }
  }

  return aBytes;

}

var SignatureValidator = {
    keyData: {"kty":"RSA","alg":"RS512","use":"sig","n":"4PyVoda4xUp2Yslx9pe3-hhOn_pr-QPyIVgJ3cKPDd6ZUgmax1pD9kEEbAfaNGcc1aFg399jDaPN_2j_ITy0kxe8q5XUlT680q5BJYV8XBFKQfKIYUcusG7MzCECjKrOOkC--aWK2JIYQWWDukS44hvxXTO78h82QE0uBDGY-qM","e":"AQAB"},

    // Returns as result to then(function(result)) true or false
    isMessageSignatureValid_promise: function(message, signature_base64) {
        return this._importKey_promise()
               .then(this._verify_promise.bind(this, message, signature_base64))
               .catch(function(err) {
                   alert("Something went wrong verifying the License Key:\n\n" + err.message + "\n" + err.stack);
               });
    },

    _verify_promise: function(message, signature_base64, key) {
        var signature       = _base64ToArrayBuffer( signature_base64 );
        var plaintext       = strToUTF8Arr( message );

        return crypto.subtle.verify(
            {name: "RSASSA-PKCS1-v1_5",  hash: { name: "SHA-512" } },
            key,
            signature,
            plaintext
        );
    },

    _importKey_promise: function() {
        return  crypto.subtle.importKey(  "jwk", //json web key
                                                this.keyData,
                                                { name: "RSASSA-PKCS1-v1_5",  hash: { name: "SHA-512" } },
                                                true,   // can extract it later if we want,
                                                ["verify"]);
    }

};