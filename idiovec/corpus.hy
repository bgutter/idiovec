;;
;; corpus.hy
;;

(import [pandas :as pd])
(import nltk)
(import [collections [Counter]])

(defclass Corpus [object]
  """
  Manages all texts for authors.
  """

  (defn --init-- [self]
    """
    Initialize a new Corpus.

    - texts: All values are passed to self.ingest
    """
    (setv self.texts (pd.DataFrame :columns ["author" "text" "word-freq" "word-len-freq"])))

  (defn ingest-dataframe [self df &optional author-col text-col]
    """
    Ingest text from a pandas DataFrame

    - df:         The DataFrame.
    - author-col: Column containing author identifier.
    - text-col:   Column containing the text data.

    If df contains only two columns, and author-col and text-col are omitted,
    then the first column will be treated as author identifiers and the second
    will be treated as text data.
    """
    (or author-col (setv author-col (first df.columns)))
    (or text-col   (setv text-col   (second df.columns)))
    (setv df (get df [author-col text-col]))
    (setv df.columns ["author" "text"])
    (setv self.texts (pd.concat [self.texts df] :ignore-index True))
    (self.--process-new-text))

  (defn ingest-dict [self texts &optional [author-key "author"] [text-key "text"]]
    """
    Ingest text from a dict.

    - texts:      Dict containing author and text as values, or a collection of them.
    - author-key: Dict key to use when accessing author info. Defaults to 'author'.
    - text-key:   Dict key to use when accessing text info. Defaults to 'text'.
    """
    (unless (instance? dict texts)
      (setv texts [texts]))
    (setv texts (dfor [author text] (texts.items) [author [text]]))
    (self.ingest-dataframe
      (pd.DataFrame.from-records texts)))

  (defn author-info [self author]
    """
    Return a DataFrame summary of the metadata accumulated for a single author.

    - author: The author to describe.
    """
    (setv rows (get self.texts.loc (= (get self.texts "author") author)))
    (-> (get rows ["word-freq" "word-len-freq"]) (.sum) (.rename author)))

  (defn --repr-- [self]
    """
    Print Corpus object.
    """
    "#Corpus")

  (defn --process-new-text [self]
    """
    Fill in NaN values in newly updated self.texts.
    """
    (setv self.texts
          (self.texts.apply
            (fn [x]
              (when (pd.isna (get x "word-freq"))
                (setv (get x "word-freq")
                      (self.--get-word-freq (get x "text"))))
              (when (pd.isna (get x "word-len-freq"))
                (setv (get x "word-len-freq")
                      (self.--get-word-len-freq (get x "word-freq"))))
              x)
            :axis 1)))

  (defn --get-word-freq [self text-str]
    """
    Generate a frequency distribution of words, given text.
    """
    (Counter (nltk.word_tokenize text-str)))

  (defn --get-word-len-freq [self word-freq]
    """
    Create a frequency distribution of word lengths, given a frequency distribution
    of words.
    """
    (Counter (dfor x (-> (dict word-freq) (.items))
                   [(len (get x 0)) (get x 1)])))

  (defn fit [self]
    """
    Fit on the data we have so far.
    """
    None))

;; debug driver
(defmain [&rest args]

  ;; Read data
  (import pickle)
  (import [matplotlib.pyplot :as plt])
  (with [fin (open "../org/programming-historian-stylometry-intro-python/comments_df.pickle" "rb")]
    (setv comment_dfs (pickle.load fin)))

  ;; Ingest data
  (setv corpus (Corpus))
  (for [df (comment_dfs.values)]
    (corpus.ingest-dataframe df "author" "body"))
  (print)
  (print corpus.texts)
  (print)
  (setv stats (get (corpus.author-info "GallowBoob") "word-len-freq"))
  (plt.figure)
  (plt.scatter (stats.keys) (stats.values))
  (plt.show))
