## proposa Italian-Japanese translator with LSTM 

I found Italian-Japanese Corpus named JAICO[1], ITADICT[2], multilingual parallel corpus from TED talks[3] on this website:

[1]JAICO https://www2.ninjal.ac.jp/past-events/2009_2021/event/specialists/project-meeting/files/JCLWorkshop_no6_papers/JCLWorkshop_No6_26.pdf 
[2]a4edu https://a4edu.unive.it/ita/index#do 

[3] https://github.com/ajinkyakulkarni14/TED-Multilingual-Parallel-Corpus


Right now, I choose [3]multilingual parallel-corpus 


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

#### problem of this corpus
there are significant missalignment in this corpus.
One of the reason for this problem is duplication of the sentence. For example, sometimes this corpus repeat same phrases like "E questo è il piano per i tetti della città.", "E questo è il piano per i tetti della città."
This phenomenon was observed particulary in japanese corpus. 

#### solution
at least, I delited the repeated part of each corpus. 

delete_duplicates/it_sentences_no_duplicates.txt: 349048→346929
delete_duplicates/ja_sentences_no_duplicates.txt: 389764→384363


