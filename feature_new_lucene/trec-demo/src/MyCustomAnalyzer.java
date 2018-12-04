import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.apache.lucene.util.Version;
import org.apache.lucene.analysis.Analyzer;

import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.core.StopAnalyzer;
import org.apache.lucene.analysis.core.StopFilter;
import org.apache.lucene.analysis.core.WhitespaceTokenizer;
import org.apache.lucene.analysis.en.PorterStemFilter;


import java.io.IOException;
import java.io.StringReader;

public class MyCustomAnalyzer extends Analyzer {

    //https://stackoverflow.com/questions/38682588/extending-lucene-analyzer
    //https://stackoverflow.com/questions/31957986/how-to-combine-analyzer-instances-for-stop-word-removal-and-stemming-in-lucene

    protected TokenStreamComponents createComponents(String s) {
        StringReader reader = new StringReader(s);
        final Tokenizer whitespaceTokenizer = new WhitespaceTokenizer();
	whitespaceTokenizer.setReader(reader);

	TokenStream tokenStream = new StopFilter(whitespaceTokenizer, StopAnalyzer.ENGLISH_STOP_WORDS_SET);
	tokenStream = new PorterStemFilter(tokenStream);
        return new TokenStreamComponents(whitespaceTokenizer, tokenStream);
    }
}
