;;
;; corpus.hy
;;

(import [pandas :as pd])
(import nltk)
(import sklearn.model-selection)
(import [collections [Counter]])

(defclass Corpus [object]
  """
  Manages all texts for authors.
  """

  (defn --init-- [self &optional raw]
    """
    Initialize a new Corpus.

    - raw: Initialize self.texts with this table. Update triggers are *not* executed.
    """
    (lif raw
        (setv self.texts raw)
        (setv self.texts (pd.DataFrame :columns ["author" "text" "word-freq" "word-len-freq"]))))

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
    (setv self.texts (pd.concat [self.texts df] :ignore-index True :sort False))
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

  (defn split-authors [self]
    """
    Return a dictionary mapping author to a Corpus of their texts.
    """
    (dfor author (self.authors)
          [author (Corpus :raw (get self.texts.loc (= (get self.texts "author") author)))]))

  (defn sample [self per-author-cnt &optional n-folds]
    """
    Return a new Corpus with per-author-cnt random texts per author.
    If an author has fewer than that many texts, they will have no
    texts in the sampled Corpus.

    - per-author-cnt: Number of texts to sample per-author
    - n-folds:        Number of Corpi to sample. If this is supplied, the
                      return value is a list of Corpus. No text will be included
                      in more than one returned Corpus. Therefore, you must have at
                      least per-author-cnt * n-folds texts. No data for authors
                      with fewer than this many texts will be included.
    """
    (setv first-sample-count (* per-author-cnt (or n-folds 1))
          author-df-map      (dfor [author corpus] (-> (self.split-authors) (.items)) [author corpus.texts])
          author-df-map      (dfor [author df] (author-df-map.items)
                                   :if (>= (len df) first-sample-count)
                                   [author (df.sample first-sample-count)]))
    (lif n-folds
         (do
           (setv kf               (sklearn.model-selection.KFold :n-splits n-folds :shuffle True)
                 author-df-splits (dfor [author df] (author-df-map.items)
                                            [author (list (kf.split df))]))
           (lfor fold-index (range n-folds)
                 (Corpus :raw
                         (pd.concat
                           (lfor [author df] (author-df-map.items)
                                 (get (. df iloc) (get author-df-splits author fold-index 1)))
                           :ignore-index True
                           :sort False))))
         (Corpus :raw
                 (pd.concat
                   (author-df-map.values)
                   :ignore-index True
                   :sort False))))

  (property
    (defn authors [self]
      """
      List all authors in this corpus.
      """
      (-> (get self.texts "author") (.unique))))

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
    (-> "#Corpus\n{}" (.format (get self.texts ["author" "text"]))))

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
                   [(len (get x 0)) (get x 1)]))))
