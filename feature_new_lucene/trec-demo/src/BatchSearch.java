import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.nio.file.Paths;
import java.lang.Float;
import java.lang.Integer;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.similarities.*;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.Version;

/** Simple command-line based search demo. */
public class BatchSearch {

	private BatchSearch() {}

	/** Simple command-line based search demo. */
	public static void main(String[] args) throws Exception {
		String usage =
				"Usage:\tjava BatchSearch [-index dir] [-simfn similarity] [-field f] [-stem] [-conjunction] [-queries file]";
		if (args.length > 0 && ("-h".equals(args[0]) || "-help".equals(args[0]))) {
			System.out.println(usage);
			System.out.println("Supported similarity functions:\ndefault: DefaultSimilary (tfidf)\n");
			System.exit(0);
		}

		String index = "index";
		String field = "contents";
		String queries = null;
		String simstring = "default";
		boolean stem = false;
		boolean conjunction = false;
		float k1 = (float)1.2; // default
		float b = (float) 0.75; // default
		int i1 = 2; // default
		int i2 = 0; // default
		int i3 = 1; // default
		boolean report = false;

		BasicModel p1[] = {new BasicModelBE(), new BasicModelG(),
				   new BasicModelP(), new BasicModelD(), new BasicModelIn(),
				   new BasicModelIne()};
		AfterEffect p2[] = {new AfterEffectL(), new AfterEffectB()};
		Normalization p3[] = {new NormalizationH1(), new NormalizationH2(), new NormalizationH3(), new NormalizationZ()};

		for(int i = 0;i < args.length;i++) {
			if ("-index".equals(args[i])) {
				index = args[i+1];
				i++;
			} else if ("-field".equals(args[i])) {
				field = args[i+1];
				i++;
			} else if ("-queries".equals(args[i])) {
				queries = args[i+1];
				i++;
			} else if ("-stem".equals(args[i])) {
			    stem = true;
			} else if ("-conjunction".equals(args[i])) {
			    conjunction = true;
			} else if ("-report".equals(args[i])) {
			    report = true;
			} else if ("-simfnGrid".equals(args[i])) {
				simstring = args[i+1];
				i+=2;
				if("bm25".equals(simstring)){
				    k1 = Float.parseFloat(args[i++]);
				    b = Float.parseFloat(args[i++]);
				}
				else if("dfr".equals(simstring)) {
				    //System.out.println(simstring+" "+args[i]);
				    i1 = Integer.parseInt(args[i++]);
				    i2 = Integer.parseInt(args[i++]);
				    i3 = Integer.parseInt(args[i++]);
				}
			} else if ("-simfn".equals(args[i])) {
				simstring = args[i+1];
				i++;
			}
		}

		Similarity simfn = null;
		if ("default".equals(simstring)) {
			simfn = new ClassicSimilarity();
		} else if ("bm25".equals(simstring)) {
		    simfn = new BM25Similarity(k1,b);
		} else if ("dfr".equals(simstring)) {
			simfn = new DFRSimilarity(p1[i1], p2[i2], p3[i3]);
		} else if ("lm".equals(simstring)) {
			simfn = new LMDirichletSimilarity();
		}
		if (simfn == null) {
			System.out.println(usage);
			System.out.println("Supported similarity functions:\ndefault: DefaultSimilary (tfidf)");
			System.out.println("bm25: BM25Similarity (standard parameters)");
			System.out.println("dfr: Divergence from Randomness model (PL2 variant)");
			System.out.println("lm: Language model, Dirichlet smoothing");
			System.exit(0);
		}
		
		IndexReader reader = DirectoryReader.open(FSDirectory.open(Paths.get(index)));
		IndexSearcher searcher = new IndexSearcher(reader);
		searcher.setSimilarity(simfn);

		Analyzer analyzer;
		if (stem){
		    // stem and stopword removal
		    analyzer = new MyCustomAnalyzer();
		}
		else{
		    analyzer = new StandardAnalyzer();
		}
		
		BufferedReader in = null;
		if (queries != null) {
			in = new BufferedReader(new InputStreamReader(new FileInputStream(queries), "UTF-8"));
		} else {
			in = new BufferedReader(new InputStreamReader(new FileInputStream("queries"), "UTF-8"));
		}
		QueryParser parser = new QueryParser(field, analyzer);
		if(conjunction){
		    // for conjunctive queries
		    parser.setDefaultOperator(QueryParser.Operator.AND);
		}

		long elaspedTime = 0;
		long ctr = 0;
		while (true) {
			String line = in.readLine();

			if (line == null || line.length() == -1) {
				break;
			}

			line = line.trim();
			if (line.length() == 0) {
				break;
			}
			
			String[] pair = line.split(" ", 2);
			Query query = parser.parse(pair[1]);
			ctr = ctr + 1;
			long startTime = System.currentTimeMillis();
			doBatchSearch(in, searcher, pair[0], query, simstring, report);
			long stopTime = System.currentTimeMillis();
			elaspedTime = elaspedTime + (stopTime - startTime);
			
		}
		double avgQueryTime = elaspedTime/(1.0*ctr);
		if(report)
		    System.out.println("Average query processing time: "+avgQueryTime+"ms");
		reader.close();
		
	}

	/**
	 * This function performs a top-1000 search for the query as a basic TREC run.
	 */
    public static void doBatchSearch(BufferedReader in, IndexSearcher searcher, String qid, Query query, String runtag, boolean report)	 
			throws IOException {

		// Collect enough docs to show 5 pages
		TopDocs results = searcher.search(query, 140646);
		ScoreDoc[] hits = results.scoreDocs;
		
		HashMap<String, String> seen = new HashMap<String, String>(140646);
		int numTotalHits = (int) results.totalHits;
		
		int start = 0;
		int end = Math.min(numTotalHits, 140646);

		if (numTotalHits != 0)
			for (int i = start; i < end; i++) {
					Document doc = searcher.doc(hits[i].doc);
					String docno = doc.get("docno");
					// There are duplicate document numbers in the FR collection, so only output a given
					// docno once.
					if (seen.containsKey(docno)) {
						continue;
					}
					seen.put(docno, docno);
					
					if(!report){
						System.out.println(qid+" Q0 "+docno+" "+i+" "+hits[i].score+" "+runtag);
					}
			}
		else{
			if(!report){
				System.out.println(qid + " None");
			}
		}
	}
}

