import numpy as np

class konvolusyon:
    def _zero_padding(self,grt,padding_boyutu):
        #padding_boyutu is padding shape.(y pad , x pad)
        (y_pad, x_pad)=padding_boyutu
        return np.pad(grt, ((y_pad, y_pad), (x_pad, x_pad), (0, 0)), 'constant', constant_values=0)

    def _padding(self,grt,padding_boyutu,padding_yontemi='zero'):
        #padding_boyutu=padding shape
        #padding_yontemi=padding technique
        if padding_yontemi=='zero':
            return self._zero_padding(grt,padding_boyutu)
        return 1

    # how we do padding for the shape we want
    def padding_boyutu_hesaplama(self,girdi_boyutu,cikti_boyutu,filtre_boyutu,kaydirma):#olmasini istedigimiz boyut icin ne kadar padding yapilmali
        # girdi_boyutu=input shape ,cikti_boyutu=output shape,filtre_boyutu=filter shape,kaydirma=stride
        #p=(((o-1)*s)+f-i)/2
        boy_pad = (((cikti_boyutu[0] - 1) * kaydirma[0]) + filtre_boyutu[0] - girdi_boyutu[0]) / 2
        en_pad = (((cikti_boyutu[1] - 1) * kaydirma[1]) + filtre_boyutu[1] - girdi_boyutu[1]) / 2
        return (int(boy_pad), int(en_pad))

    #shape calculation for after convolution
    def konvolusyon_sonrasi_olusacak_boyut_hesabi(self,goruntu_boyutu,filtre_boyutu,kaydirma,padding=(0,0)):
        #((g-f+2*p)/k)+1=c , ((i-f+2*p)/s)+1=o
        yeni_boy = ((goruntu_boyutu[0] - filtre_boyutu[0] + 2 * padding[0]) / kaydirma[0]) + 1
        yeni_en = ((goruntu_boyutu[1] - filtre_boyutu[1] + 2 * padding[1]) / kaydirma[1]) + 1
        return (int(yeni_boy), int(yeni_en))

    #padding to keep the image the same shape
    def ayni_boyut_icin_padding(self,grt,filtre_boyutu,kaydirma=(1,1),padding_yontemi='zero'):
        boyut=grt.shape[0],grt.shape[1]
        padding_boyutu=self.padding_boyutu_hesaplama(boyut,boyut,filtre_boyutu,kaydirma)
        return self._padding(grt,padding_boyutu,padding_yontemi)

    #convolution for gray scale image
    def _konvolusyon_gri(self,grt,filtre,kaydirma=(1,1),padding=False,aktivasyon_fonksiyonu='relu',biases=None):#tam kontrol yapilmadi!!!!
        #girdi=input,filtre=filter,kaydirma=stride(tuple(x,y)),if padding is True input shape = output shape
        ksob=self.konvolusyon_sonrasi_olusacak_boyut_hesabi(grt.shape,filtre.shape, kaydirma)
        if padding==True:
            yeni=self.ayni_boyut_icin_padding(grt,filtre.shape,kaydirma,'zero')
        else:
            yeni=np.zeros(ksob,dtype="float32")

        goruntu_boy_bitis = (kaydirma[0]*(ksob[0]-1))+1
        goruntu_en_bitis = (kaydirma[1]*(ksob[1]-1))+1
        yeni_boy_index=0
        for boy in range(0,goruntu_boy_bitis,kaydirma[0]):
            yeni_en_index=0
            for en in range(0,goruntu_en_bitis,kaydirma[1]):
                deger=np.sum(np.multiply(grt[boy:boy+filtre.shape[0],en:en+filtre.shape[1]],filtre))
                yeni[yeni_boy_index][yeni_en_index]=deger
                yeni_en_index+=1
            yeni_boy_index+=1

        return yeni

    #concatenating r g b channels
    def _rgb_kanallari_birlestir(self,b,g,r,veri_tipi="float32"):
        yeni = np.zeros((b.shape[0], b.shape[1], 3), dtype=veri_tipi)
        for i in range(b.shape[0]):
            for j in range(b.shape[1]):
                yeni[i][j][0] = b[i][j]
                yeni[i][j][1] = g[i][j]
                yeni[i][j][2] = r[i][j]
        return yeni

    # convolution for rgb image
    def _konvolusyon_rgb(self, grt, filtre, kaydirma=(1,1), padding=False,aktivasyon_fonksiyonu='relu', biases=None):
        b = self._konvolusyon_gri(grt[:, :, 0], filtre, kaydirma, padding, aktivasyon_fonksiyonu, biases=None)
        g = self._konvolusyon_gri(grt[:, :, 1], filtre, kaydirma, padding, aktivasyon_fonksiyonu, biases=None)
        r = self._konvolusyon_gri(grt[:, :, 2], filtre, kaydirma, padding, aktivasyon_fonksiyonu, biases=None)
        return self._rgb_kanallari_birlestir(b,g,r)

    # convolution
    def konvolusyon_islemi(self, grt, filtre, kaydirma=(1,1), padding=False, aktivasyon_fonksiyonu='relu', biases=None):
        if len(grt.shape)==3:
            return self._konvolusyon_rgb(grt, filtre, kaydirma, padding, aktivasyon_fonksiyonu, biases=None)
        else:
            return self._konvolusyon_gri(grt, filtre, kaydirma, padding, aktivasyon_fonksiyonu, biases=None)
