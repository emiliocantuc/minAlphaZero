# minAlphaZero

Yet another AlphaZero toy implementation for educational purposes. 
There are many good ones already -- this one is just for my understanding.

Stuff missing:
- fix bugs & cleanup
- board symmetries

A few details that surprised me or are not that obvious:
- don't train the network only on the current episode's examples; keep a buffer to not overfit
