from dataclasses import dataclass, asdict
from .misc import Color
def ndarray(dims, fill=0):
    if(len(dims) == 1):
        n = dims[0]
        return [fill for i in range(n)]
    else:
        return [ndarray(dims[1:], fill=fill) for i in range(dims[0])]
def element_similarity(a, b):
    if(a.lower()==b.lower()):
        return 1
    else:
        return 0
def colored(st, fore=None, back=None):
    ls = []
    if(fore is not None):
        ls.append(Color.from_any(fore).as_terminal_fg())
    if(back is not None):
        ls.append(Color.from_any(back).as_terminal_bg())
    ls.append(st)
    if(fore or back):
        ls.append(Color().as_terminal_rst())
    return "".join(ls)



@dataclass
class _lcs:
    A: str
    B: str
    common: str
    common_ratio_a: float
    common_ratio_b: float
    common_ratio: float
    common_len: int
    a_matched: list
    b_matched: list
    def calc1(A, B, f_similarity = element_similarity):
        sa = set(A)
        sb = set(B)
        a_matched = ndarray((len(A),), False)
        b_matched = ndarray((len(B),), False)
        simple_a = list()
        simple_b = list()
        for idx, i in enumerate(A):
            if(i in sb):
                simple_a.append((idx, i))
        for idx, i in enumerate(B):
            if(i in sa):
                simple_b.append((idx, i))
        n = len(simple_a)
        m = len(simple_b)
        dp = ndarray((n, m), 0)
        dp_from = ndarray((n, m), (-1, -1))
        for i, a in enumerate(simple_a):
            for j, b in enumerate(simple_b):
                _dp = 0
                _from = (-1, -1)
                sim = f_similarity(a[1], b[1])
                if(sim):
                    __dp = (dp[i-1][j-1] if (i and j) else 0)+sim
                    __from = i-1, j-1
                    if(__dp>_dp):
                        _dp = __dp
                        _from = __from
                if(i):
                    __dp = dp[i-1][j]
                    __from = (i-1, j)
                    if(__dp>_dp):
                        _dp = __dp
                        _from = __from
                if(j):
                    __dp = dp[i][j-1]
                    __from = (i, j-1)
                    if(__dp>_dp):
                        _dp = __dp
                        _from = __from
                dp[i][j]=_dp
                dp_from[i][j]=_from
        
        common = []
        u, v = n-1, m-1
        while(u!=-1 and v!=-1):
            u1, v1 = dp_from[u][v]
            if(u1 == u-1 and v1 == v-1):
                adx, a = simple_a[u]
                bdx, b = simple_b[v]
                a_matched[adx]=True
                b_matched[bdx]=True
                common.append(a)
            u,v = u1, v1
        # print(dp)
        if(n and m):
            common_len = dp[n-1][m-1]
        else:
            common_len = 0
        common_ratio_a = common_len/(len(A)+1e-10)
        common_ratio_b = common_len/(len(B)+1e-10)
        common_ratio = common_ratio_a*common_ratio_b
        return _lcs(A, B, common[::-1], common_ratio_a, common_ratio_b, common_ratio, common_len, a_matched, b_matched)
    def calc(A, B, weights = None):
        global _debug
        if(weights is not None):
            weights = dict()
        n = len(A)
        m = len(B)
        dp = ndarray((n, m))
        a_matched = ndarray((n,), False)
        b_matched = ndarray((m,), False)
        dp_from = ndarray((n, m), (-1, -1))
        sb = set(B)
        for i in range(n):
            if(i and (A[i] not in sb)):
                dp[i]=dp[i-1]
                dp_from[i]=dp_from[i-1]
                continue
            for j in range(m):
                '''if(A[i] in 'Aa' and B[j] in 'Aa' and A[i]!=B[j]):
                    print(A[i],B[j],A[i].lower() == B[j].lower())'''
                mx = 0
                _dp_from = (-1, -1)

                # match A[i], B[j]
                score = dp[i-1][j-1] if (i and j) else 0
                score1 = element_similarity(A[i], B[j])

                if(score1):

                    if(score+score1 >= mx):
                        mx = score+score1
                        _dp_from = (i-1, j-1)
                if(i):
                    if(dp[i-1][j] >= mx):
                        mx = dp[i-1][j]
                        _dp_from = (i-1, j)
                if(j):
                    if(dp[i][j-1] >= mx):
                        mx = dp[i][j-1]
                        _dp_from = (i, j-1)
                dp[i][j] = mx
                dp_from[i][j] = _dp_from
        u, v = n-1, m-1
        common = []
        while(u >= 0 and v >= 0):
            
            u1, v1 = dp_from[u][v]
            # print(u, v, u1, v1)
            ''' if(_debug):
                print(u, v, 'from', u1, v1) '''
            if(u1 == u-1 and v1 == v-1):
                if(element_similarity(A[u], B[v]) > 0.5):
                    common.append(A[u])
                    '''if(_debug):
                        print("matching", A[u], B[v])'''
                    a_matched[u] = True
                    b_matched[v] = True
            u, v = u1, v1

        common = common[::-1]
        '''#self.A = A
        self.B = B
        self.common = common  # list
        common_ratio_a = dp[n-1][m-1]/len(A)
        common_ratio_b = dp[n-1][m-1]/len(B)
        self.common_ratio = self.common_ratio_a*self.common_ratio_b
        self.common_len = dp[n-1][m-1]'''
        common_len = dp[n-1][m-1]
        common_ratio_a = common_len/len(A)
        common_ratio_b = common_len/len(B)
        return _lcs(A, B, common, common_ratio_a, common_ratio_b, common_ratio_a*common_ratio_b, common_len, a_matched, b_matched)

    def color_common(self, foreA="RED", foreB="GREEN"):
        retA = []
        for idx, i in enumerate(self.A):
            if(self.a_matched[idx]):
                retA.append(str(colored(i, fore=foreA)))
            else:
                retA.append(i)
        retA = "".join(retA)

        retB = []
        for idx, i in enumerate(self.B):
            if(self.b_matched[idx]):
                retB.append(str(colored(i, fore=foreB)))
            else:
                retB.append(i)
        retB = "".join(retB)
        return retA, retB

    def asdict(self, preserve_AB=False):
        D = asdict(self)
        if(not preserve_AB):
            D.pop("A")
            D.pop("B")
        D["a_matched"] = "".join(['1' if i else '0' for i in self.a_matched])
        D["b_matched"] = "".join(['1' if i else '0' for i in self.b_matched])
        return D

    def fromdict_A_B(D, A, B):
        D['a_matched'] = [bool(int(i)) for i in D["a_matched"]]
        D['b_matched'] = [bool(int(i)) for i in D["b_matched"]]
        return _lcs(A=A, B=B, **D)

def LCS(A, B):
    
    return _lcs.calc1(A, B)
if(__name__=="__main__"):
    A = "1girl, highres, pink hair, absurdres, ".split(",")
    B = "1girl, highres, absurdres, red eyes".split(",")
    l = LCS(A, [])
    print(*l.color_common())
    print(l.common_len)
    print(l.a_matched)
    print(l.b_matched)
    print(l.common)