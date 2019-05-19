;;
;; debug.hy
;;
;; Debugging code.
;;

(import [corpus [*]])

;; debug driver
(defmain [&rest args]

  ;; Read data
  (import pickle)
  (import [matplotlib.pyplot :as plt])
  (with [fin (open "../../org/programming-historian-stylometry-intro-python/comments_df.pickle" "rb")]
    (setv comment_dfs (pickle.load fin)))

  ;; Ingest data
  (setv corpus (Corpus))
  (for [df (comment_dfs.values)]
    (corpus.ingest-dataframe df "author" "body"))
  (setv gb-corpus (get (corpus.split-authors) "GallowBoob"))
  (for [c (corpus.sample 3 6)]
    (print (c.author-info "GallowBoob")))

  ;;(plt.figure)
  ;;(for [auth (corpus.authors)]
  ;;  (setv stats (get (corpus.author-info auth) "word-len-freq"))
  ;;  (plt.scatter (stats.keys) (stats.values)))
  ;;(plt.show)
  )
