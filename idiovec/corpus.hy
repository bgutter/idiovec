;;
;; corpus.hy
;;

(defclass Corpus [object]
  """
  Manages all texts for authors.
  """

  (defn --init-- [self &optional [texts None] [author-key None]]
    """
    Initialize a new Corpus.

    - texts: An iterable of text objects
    """
    (setv self.--texts [])
    (for [t texts]
      (self.register-text t)))

  (defn --repr-- [self]
    """
    Print Corpus object.
    """
    (.join  "\n\t"
            (+ ["#Corpus:"]
              (lfor x (cut self.--texts 0 (min 5 (len self.--texts)))
                  (cut (x.lower) 0 (min 50 (len x)))))))

  (defn register-text [self text]
    """
    Register a new text into the corpus.

    - text: A text object.
    """
    (.append self.--texts text))

  (defn fit [self]
    """
    Fit on the data we have so far.
    """
    None))

;; debug driver
(defmain [&rest args]
  (setv corpus (Corpus (cut args 1)))
  (corpus.fit)
  (print corpus))
