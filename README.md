## proposa Italian-Japanese translator with LSTM 

### repository which I will use for the base line:

1. deeplearning from scratch(Oreilly)
- https://github.com/oreilly-japan/deep-learning-from-scratch-2 
- https://github.com/naoya526/Deeplearning2
2. Pavia University "Machine Learning (Professor Claudio Cusano)"
- English_to_Italian_automatic_translation.ipynb

### My Goal and What to do
1. Morphological analysis
- When it comes to process non-Alphabet language (Alabic, Japanese, Chinese, Corean), you need to do **Morphological analysis**. I'm thinking to use tool, "Vibrato" for this process.
In japanese, there's no space for divide words like English. ItmeansJapansewritelikethis.
Hence, It is requrired to divide into each sentences, so it could be classified as one process in the sequence of the tokenization.


### dataset 
I found Italian-Japanese Corpus named JAICO[1], ITADICT[2], multilingual parallel corpus from TED talks[3] on these website:

1. **JAICO** https://www2.ninjal.ac.jp/past-events/2009_2021/event/specialists/project-meeting/files/JCLWorkshop_no6_papers/JCLWorkshop_No6_26.pdf 
2. **a4edu** https://a4edu.unive.it/ita/index#do 
3. **TED-Multilingual-Parallel-Corpus** https://github.com/ajinkyakulkarni14/TED-Multilingual-Parallel-Corpus


Right now, I'm thinking to use  [3]multilingual parallel-corpus


### comparison between italian and japanese
```italian
Line 3664454 (#349044): E questo è il piano per i tetti della città.
Line 3664464 (#349045): Abbiamo sopraelevato la terra sui tetti.
Line 3664475 (#349046): Gli agricoltori hanno piccoli ponti per circolare da un tetto all'altro.
Line 3664486 (#349047): Occupiamo la città con spazi abitativi e lavorativi in tutti i piani terra.
Line 3664497 (#349048): Quindi, questa è la città esistente, e questa è la città nuova.
```
row: 346929

```japanese
Line 1654306 (#389760): 従来の土壌を屋根の上に持ち上げ
Line 1654310 (#389761): 農業者は屋根から屋根へと
Line 1654315 (#389762): 一階部分は仕事と生活のための
Line 1654320 (#389763): これが現在の街で こちらが新しい街です
Line 1654324 (#389764): （拍手）
```
row: 389764

#349048 Quindi, questa è la città esistente, e questa è la città nuova.
corresponds to 
#389763 これが現在の街で こちらが新しい街です
at least these last sentence shows same meaning.

#### problem of this corpus
there are significant missalignment in this corpus.
One of the reason for this problem is duplication of the sentence. For example, sometimes this corpus repeat same phrases like "E questo è il piano per i tetti della città.", "E questo è il piano per i tetti della città."
This phenomenon was observed particulary in japanese corpus. 

#### solution
at least, I deleted the repeated part of each corpus and Split them into Batch which contain 1,000 row.

delete_duplicates/it_sentences_no_duplicates.txt: 349048→346929
delete_duplicates/ja_sentences_no_duplicates.txt: 389764→384363

Still there are significant missalignment, so I'm still thinking how to process this dataset. 

1. Choose another dataset - this could be option. Correcting missalignment of 40,000 row is not realistic. There might be English-multilingual Dataset.(Eng-Jpn,Eng-Ita)
2. Invastingating with 500 row, if I detect the miss alignment, just delete that exessive part of corpus and align properly. This could be option. the problem is also requiring a lot of time and petience.
3. Choose another languages which has more option for the multilingual dataset : We don't have Italian-Japanese multilingual dataset. For example, I can try with French-Italian dataset, but I think if we don't have enough dataset, it means the model could be valuable.




