def my_dispersion_plot(text, words, ignore_case=False):

    try:
        import numpy as np
        from PIL import Image
        from matplotlib import pyplot as plt
    except ImportError:
        raise ValueError('The plot function requires the matplotlib, PIL and numpy packages.')

    text = list(text)
    words.reverse()

    
    if ignore_case:
        text_to_comp = list(map(str.lower, text))
        words_to_comp = list(map(str.lower, words))
    else:
        text_to_comp = text
        words_to_comp = words
        
    points = [(x, y) for x in range(len(text_to_comp))
                     for y in range(len(words_to_comp))
                     if text_to_comp[x] == words_to_comp[y]]
    
    if points:
        x, y = list(zip(*points))

    else:
        x = y = ()
   
    # img = np.array(Image.open('Abe0.jpg'))
    # plt.imshow(img, zorder=0, extent=[0, img.shape[1]*7, 0, 5], alpha=0.3)
    plt.plot(x, y, "k|", scalex=.5)
    plt.yticks(list(range(len(words))), words, fontsize=12, color="k")
    # plt.yticks([y*img.shape[0]*7/(len(words)+1) for y in range(1, len(words)+1)], words, fontsize=12, color="k")
    plt.ylim(-1, len(words))
    # plt.ylim(0, img.shape[0]*7)
    plt.title("Lexical Dispersion Plot", fontsize=15) #, color = '#')
    plt.xlabel("Word Offset", fontsize=12)
    
    plt.savefig('dispersion_plot.png', transparent = True, dpi = 300, bbox_inches='tight')
    plt.show()
