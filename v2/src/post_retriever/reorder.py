class LongContextReorder:

    @staticmethod
    def reorder(doc_score_pairs: list) -> list:

        if len(doc_score_pairs) <= 2:
            return doc_score_pairs
        
        
        reordered=[None]*len(doc_score_pairs)
        left=0

        right=len(doc_score_pairs)-1

        for i,item in enumerate(doc_score_pairs):
            if i%2==0:
                reordered[left]=item
                left+=1
            else:
                reordered[right]=item
                right-=1