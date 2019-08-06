"""
cruft.py

Things idiovec probably shouldn't try to do but does anyway because someone somewhere will expect
it to do so.
"""

import html
import re

def reddit_sanitize( text ):
    """
    Convert comments in the Reddit API format to actual plain-text likely
    constructed by the individual who posted it. HTML is unescaped, markup
    is removed, and quotes are removed.
    """

    # Unescape HTML (IE, '&gt;' becomes '>')
    text = html.unescape( text )

    # Remove markup
    enclosed_text_regexes = [
        re.compile( r"\*\*(\S+[^*]*\S+|\S)\*\*" ),    # Bold
        re.compile( r"\*(\S+[^*]*\S+|\S)\*" ),        # Italic
        re.compile( r"_(\S+[^_]*\S+|\S)_" ),          # Undelrine
        re.compile( r"\~\~(\S+[^\~]*\S+|\S)\~\~" ),   # Strikethrough
        re.compile( r"\>\!(\S+[^(!<)]*\S+|\S)\!\<" ), # Spoilers
        re.compile( r"\^(\S+)" ),                     # Superscript
        re.compile( r"\[([^\]]*)\]\([^\)]+\)" ),      # Links, remove link but keep text.
        ]
    for rgx in enclosed_text_regexes:
        text = re.sub( rgx, r"\1", text )

    # Remove quoted and preformatted lines
    quote_filter_pred = lambda line: len( line ) <= 0 or line[ 0 ] != ">"
    pref_filter_pred  = lambda line: ( ( len( line ) <= 4 or line[ :4 ] != "    " ) and
                                       ( len( line ) <= 0 or line[ 0 ] != "\t" ) )
    lines = text.split( "\n" )
    return "\n".join( [ x for x in lines if quote_filter_pred( x ) and pref_filter_pred( x ) ] )
