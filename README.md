# GPT Decoder
A simple GPT decoder for educational purposes following Andrej Karpathy's 'Let's build GPT: from scratch, in code, spelled out'.

Here's an example output trained on some Shakespearean text:
```
0.212801 M parameters
{'step': 0, 'train_loss': 4.4182, 'val_loss': 4.4209, 'time_elapsed': 2.3046}
{'step': 200, 'train_loss': 2.3879, 'val_loss': 2.3779, 'time_elapsed': 11.1771}
{'step': 400, 'train_loss': 2.1928, 'val_loss': 2.2064, 'time_elapsed': 20.0753}
{'step': 600, 'train_loss': 2.0966, 'val_loss': 2.1343, 'time_elapsed': 29.0722}
{'step': 800, 'train_loss': 2.0105, 'val_loss': 2.0704, 'time_elapsed': 37.9091}
{'step': 1000, 'train_loss': 1.9735, 'val_loss': 2.0389, 'time_elapsed': 46.7602}
{'step': 1200, 'train_loss': 1.9167, 'val_loss': 2.0161, 'time_elapsed': 55.6576}
{'step': 1400, 'train_loss': 1.8975, 'val_loss': 1.9847, 'time_elapsed': 64.4611}
{'step': 1600, 'train_loss': 1.8542, 'val_loss': 1.9728, 'time_elapsed': 73.3032}
{'step': 1800, 'train_loss': 1.8322, 'val_loss': 1.9692, 'time_elapsed': 82.1453}
{'step': 2000, 'train_loss': 1.8038, 'val_loss': 1.9316, 'time_elapsed': 91.0641}
{'step': 2200, 'train_loss': 1.7885, 'val_loss': 1.9243, 'time_elapsed': 99.984}
{'step': 2400, 'train_loss': 1.7681, 'val_loss': 1.9058, 'time_elapsed': 108.8979}
{'step': 2600, 'train_loss': 1.7443, 'val_loss': 1.8873, 'time_elapsed': 117.8202}
{'step': 2800, 'train_loss': 1.7274, 'val_loss': 1.87, 'time_elapsed': 126.6845}
{'step': 3000, 'train_loss': 1.7184, 'val_loss': 1.8836, 'time_elapsed': 135.5954}
{'step': 3200, 'train_loss': 1.7087, 'val_loss': 1.8718, 'time_elapsed': 144.4997}
{'step': 3400, 'train_loss': 1.6908, 'val_loss': 1.8309, 'time_elapsed': 153.4739}
{'step': 3600, 'train_loss': 1.6865, 'val_loss': 1.8365, 'time_elapsed': 162.3691}
{'step': 3800, 'train_loss': 1.6697, 'val_loss': 1.8342, 'time_elapsed': 171.085}
{'step': 4000, 'train_loss': 1.6739, 'val_loss': 1.8452, 'time_elapsed': 180.0184}
{'step': 4200, 'train_loss': 1.6632, 'val_loss': 1.8167, 'time_elapsed': 188.9192}
{'step': 4400, 'train_loss': 1.6545, 'val_loss': 1.8161, 'time_elapsed': 197.7546}
{'step': 4600, 'train_loss': 1.6503, 'val_loss': 1.8109, 'time_elapsed': 206.7041}
{'step': 4800, 'train_loss': 1.6428, 'val_loss': 1.8002, 'time_elapsed': 215.566}
{'step': 4999, 'train_loss': 1.6204, 'val_loss': 1.7887, 'time_elapsed': 224.4521}


Clown:
Reride will time yet that her bobed and Stirn.

MARCALUS:
I'll but foul him vetter diladom.

From way.

HENRY BOLINGBROKE:
Ay, of it he me milend!

AUSERDixtend, I in latient,
Wor the hold me now on that speliling tear.
How my depiry, the hair's, why. Here, she ray, I lives
Morthy would that
To Willo when evicks to the purive and him:
A poor of his burd kindnust fiel so;
Another whit thy fled at than Preed my off.
Her such still is wards.
With Edward hope cource.

Any Irother,
And for his need sound, griefs your ganthm of a temps one not between.
Are him; stare it with the cours.

TYBRTHUMIO:
Bestatchs lord
Is as he lord.

Serve everer:
Well! to did the Leavos in meen; boy
Of the husbrea e'er as not come,
Against is him it, I do, soxtent.

COMINIUS:
O greparing weal so! with unlower,
Whose lady, 'tis your speding,--

Thixderd.

CLARIO:
Let thief seoff, to me, God? Bonour ce summancisort,
These and growth our speakless king I say and reff,
Are at hotived out it forcley the gion;
The can it we softs, I hurl no whence nurch is though this do none:
Way.

Is, and men, me we minds Of tendery,
Bid it son highned of the drach
The read
Hear womenIning talk the calload, and any inkerity, with the Lord, of dead. Convisonshal yet of I is thavest outwer.H:
For tappurlels,
Have me of thee ofjer'd her wants much he call bons?

QUEEN ELIZABETH:
Were you, not thou art have be-dase.

QUEEN ELIZABETH:
Well, a beliden heaven it none to me,
Sie te you thelpievence yee of Rome of Rolth, there smost of you;
He their arm, every from me of a-latester:
O there, to if these world, she revere wall prever him kind on the hopp.

BENCUS:
O, marry, no have the sur, give affetch
forher, byor some not hither.

COMINIUS:
When: he's call, good many,
I corpous cilt, of montles.

MENENIUS:
Ay here!

BRIZTHIO:
I' I have but I lip' it, i'like he putile
I door to him would the Lold, say, try promind thy awonsent,
But hath and will on the Riparcond:
In it shown'd fool thee such come a her be our are
```

## Optimisations

### Improving Transformer Optimization Through Better Initialization Paper
- Added Xavier Uniform Init.
- Added T-FixUp Init.

### Primer: Searching for Efficient Transformers for Language Modeling Paper
- Added SquaredReLU.
- Added Depthwise Convolution.
