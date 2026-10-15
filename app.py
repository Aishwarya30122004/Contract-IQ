import io, re
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "ContractIQ_clause_dataset.csv"
st.set_page_config(page_title="ContractIQ Portal", page_icon="📑", layout="wide")

st.markdown("""
<style>
.block-container{max-width:1450px;padding-top:1.2rem}
[data-testid="stAppViewContainer"]{background:#f4f7fb}
[data-testid="stSidebar"]{background:#111c30}
[data-testid="stSidebar"] *{color:#edf3ff!important}
.hero{padding:30px;border-radius:22px;color:white;background:linear-gradient(115deg,#111c30,#1d4b89,#3979dc);margin-bottom:18px}
.hero h1{color:white;font-size:2.6rem;margin:0}.hero p{color:#e5edff;font-size:1.05rem}
div[data-testid="stMetric"]{background:white;border:1px solid #e3e9f2;padding:14px;border-radius:15px}
.card{background:white;border:1px solid #e3e9f2;border-radius:16px;padding:18px}
</style>
""", unsafe_allow_html=True)

def clean(s): return re.sub(r"\s+", " ", str(s or "").lower()).strip()

def extract(file):
    raw, name = file.getvalue(), file.name.lower()
    if name.endswith(".txt"): return raw.decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        return "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(raw)).pages)
    if name.endswith(".docx"):
        from docx import Document
        d = Document(io.BytesIO(raw))
        return "\n".join([p.text for p in d.paragraphs if p.text.strip()])
    raise ValueError("Upload PDF, DOCX or TXT.")

def split(text):
    parts = [re.sub(r"\s+", " ", x).strip() for x in re.split(r"\n\s*\n", text) if x.strip()]
    if len(parts) <= 1: parts = [x.strip() for x in re.split(r"(?<=[.;])\s+(?=[A-Z0-9])", text) if x.strip()]
    return [x for x in parts if len(x) >= 20] or [x.strip() for x in text.splitlines() if len(x.strip()) >= 20]

@st.cache_resource
def models():
    if not DATA.exists(): raise FileNotFoundError("Put ContractIQ_clause_dataset.csv in data/.")
    df = pd.read_csv(DATA)
    needed = {"clause_text","clause_type","risk_level"}
    if not needed.issubset(df.columns): raise ValueError("Dataset must contain clause_text, clause_type, risk_level.")
    df = df.dropna(subset=list(needed)).copy()
    df["clause_text"] = df.clause_text.map(clean)
    out = {}
    for target in ["clause_type","risk_level"]:
        p = Pipeline([("tfidf",TfidfVectorizer(max_features=3000,ngram_range=(1,2))),("model",LogisticRegression(max_iter=1500,class_weight="balanced"))])
        p.fit(df.clause_text,df[target].astype(str)); out[target] = p
    return df,out

def retrieve(question, clauses, n=4):
    if not question.strip() or not clauses: return []
    v=TfidfVectorizer(ngram_range=(1,2),stop_words="english")
    m=v.fit_transform([clean(c) for c in clauses]+[clean(question)])
    scores=cosine_similarity(m[-1],m[:-1]).ravel()
    return [(int(i),float(scores[i]),clauses[int(i)]) for i in scores.argsort()[::-1][:n] if scores[i]>0]

def policy_findings(text, opts):
    t=clean(text); f=[]
    if opts["cap"] and any(k in t for k in ["liability","liable","damages"]) and not any(k in t for k in ["shall not exceed","limited to","liability cap","aggregate liability","maximum liability"]):
        f.append(("High","Liability cap","Liability wording found without an obvious cap.","Ask a qualified reviewer whether a clear cap and exceptions are needed."))
    if opts["days"] and any(k in t for k in ["payment","invoice","pay "]):
        nums=[int(n) for n in re.findall(r"\b(\d{1,3})\s*days?\b",t)]
        if nums and min(nums)>opts["days"]: f.append(("Medium","Payment period",f"Payment period may exceed {opts['days']} days.","Compare with company payment policy."))
        elif not nums: f.append(("Medium","Payment period","Payment deadline not detected.","Confirm the deadline is explicit."))
    if opts["renew"] and any(k in t for k in ["automatically renew","automatic renewal","auto-renew","shall renew automatically"]):
        f.append(("High","Automatic renewal","Potential automatic renewal wording detected.","Review notice window and opt-out process."))
    if opts["privacy"] and any(k in t for k in ["personal data","personal information","data processing","privacy"]) and not any(k in t for k in ["data protection","security measures","breach notification","applicable data protection laws"]):
        f.append(("Medium","Data privacy","Explicit data safeguards may need review.","Check permitted use, security, retention and breach notice."))
    if opts["term"] and "termination" in t and not any(k in t for k in ["written notice","days notice","material breach","for cause"]):
        f.append(("Medium","Termination","Notice or cause terms not detected.","Clarify triggers, notice and cure period."))
    if any(k in t for k in ["indemnify","indemnification","hold harmless"]) and any(k in t for k in ["all claims","any and all","all losses","regardless of"]):
        f.append(("High","Broad indemnification","Broad indemnification wording detected.","Review scope, exclusions and mutuality.")
        )
    return f

for k,v in {"logged_in":False,"user":"","role":"Employee","text":"","clauses":[],"results":pd.DataFrame(),"filename":"","chat":[]}.items():
    if k not in st.session_state: st.session_state[k]=v

if not st.session_state.logged_in:
    a,b=st.columns([1.15,.85],gap="large")
    with a:
        st.markdown('<div class="hero"><p>ENTERPRISE CONTRACT INTELLIGENCE</p><h1>Contracts, understood.<br>Risks, made visible.</h1><p>One workspace for employees, vendors and business partners to review contract language, explore risk signals and find answers faster.</p><br>✦ Clause intelligence &nbsp; ✦ Policy checks &nbsp; ✦ Contract assistant</div>',unsafe_allow_html=True)
        x,y,z=st.columns(3); x.metric("Clause review","AI-assisted"); y.metric("Policy checks","Configurable"); z.metric("Q&A","Evidence-based")
        st.caption("Demo prototype. Use fictional or approved test documents only.")
    with b:
        st.markdown("## Welcome to ContractIQ")
        st.write("Sign in to open your workspace.")
        st.info("Demo login only — this form does not authenticate real company accounts.")
        with st.form("login"):
            username=st.text_input("Username",placeholder="e.g. demo.employee")
            password=st.text_input("Password",type="password",placeholder="Any demo password")
            role=st.selectbox("I am a…",["Employee","Vendor","Business partner","Contract reviewer"])
            go=st.form_submit_button("Enter ContractIQ",type="primary",use_container_width=True)
        if go:
            if username.strip() and password:
                st.session_state.logged_in=True; st.session_state.user=username.strip(); st.session_state.role=role; st.rerun()
            else: st.error("Enter a username and password.")
        st.markdown('<div class="card"><b>Your workspace includes</b><br><br>📊 Dashboard<br>📄 Contract review<br>🛡️ Policy compliance<br>💬 Interactive assistant<br>📥 Exportable reports</div>',unsafe_allow_html=True)
    st.stop()

try: df,ml=models()
except Exception as e: st.error(f"Model setup failed: {e}"); st.stop()

with st.sidebar:
    st.markdown("## 📑 ContractIQ")
    st.caption("ENTERPRISE PORTAL")
    st.write("Signed in as"); st.markdown(f"**{st.session_state.user}**")
    st.caption(st.session_state.role)
    page=st.radio("WORKSPACE",["Home dashboard","Analyze a contract","AI assistant","Policy compliance","Export report","About & security"])
    st.divider()
    if st.button("Sign out",use_container_width=True):
        st.session_state.logged_in=False; st.session_state.chat=[]; st.rerun()

st.markdown(f'<div class="hero"><p>{st.session_state.role.upper()} WORKSPACE</p><h1>Welcome, {st.session_state.user.replace("<","").replace(">","")}</h1><p>Review terms, spot possible risks and ask questions with clause-level evidence.</p></div>',unsafe_allow_html=True)

if page=="Home dashboard":
    r=st.session_state.results
    total=len(r); high=int(r["Predicted risk"].astype(str).str.lower().eq("high").sum()) if total else 0
    a,b,c,d=st.columns(4); a.metric("Current contract",st.session_state.filename or "None"); b.metric("Clauses analyzed",total); c.metric("High-risk predictions",high); d.metric("Training examples",len(df))
    st.markdown("### Quick actions")
    a,b,c=st.columns(3)
    a.markdown('<div class="card"><h3>📄 Review a contract</h3><p>Extract and classify contract clauses.</p></div>',unsafe_allow_html=True)
    b.markdown('<div class="card"><h3>🛡️ Check policy</h3><p>Flag possible deviations from review rules.</p></div>',unsafe_allow_html=True)
    c.markdown('<div class="card"><h3>💬 Ask the assistant</h3><p>Find relevant contract passages and next steps.</p></div>',unsafe_allow_html=True)
    if total:
        x,y=st.columns(2)
        with x: st.plotly_chart(px.pie(r,names="Predicted risk",hole=.45,title="Risk distribution"),use_container_width=True)
        with y: st.plotly_chart(px.bar(r,x="Predicted clause type",title="Clause categories"),use_container_width=True)
    else:
        st.info("Start by choosing **Analyze a contract** in the sidebar.")
        chart=df["clause_type"].value_counts().rename_axis("Clause type").reset_index(name="Rows")
        st.plotly_chart(px.bar(chart,x="Clause type",y="Rows",title="Clause categories in training dataset"),use_container_width=True)

elif page=="Analyze a contract":
    st.markdown("### Contract review studio")
    file=st.file_uploader("Upload PDF, DOCX or TXT",type=["pdf","docx","txt"])
    if file and st.button("Analyze contract",type="primary"):
        try:
            text=extract(file); clauses=split(text)
            if not clauses: st.warning("No readable text detected. Scanned PDFs need OCR.")
            else:
                rows=[]
                for i,c in enumerate(clauses,1):
                    x=clean(c); tm=ml["clause_type"]; rm=ml["risk_level"]
                    tp=tm.predict_proba([x])[0]; rp=rm.predict_proba([x])[0]; risk=rm.predict([x])[0]
                    reasons=[]
                    for keys,msg in [(["unlimited liability","without limitation"],"Potentially uncapped liability wording"),(["all claims","any and all","regardless of fault"],"Broad responsibility or indemnification wording"),(["automatically renew","automatic renewal","auto-renew"],"Automatic renewal wording"),(["personal data","personal information"],"Personal data obligations appear")]:
                        if any(k in x for k in keys): reasons.append(msg)
                    if not reasons: reasons.append(f"Model predicted {risk}; compare wording with approved policy.")
                    rows.append({"Clause #":i,"Clause text":c,"Predicted clause type":tm.predict([x])[0],"Predicted risk":risk,"Type confidence":round(float(tp.max()),3),"Risk confidence":round(float(rp.max()),3),"Risk explanation":"; ".join(reasons)})
                st.session_state.text=text; st.session_state.clauses=clauses; st.session_state.results=pd.DataFrame(rows); st.session_state.filename=file.name; st.session_state.chat=[]
                st.success(f"Analyzed {len(rows)} candidate clauses.")
        except Exception as e: st.error(f"Could not analyze: {e}")
    r=st.session_state.results
    if not r.empty:
        a,b,c,d=st.columns(4); a.metric("Clauses",len(r)); b.metric("High",int(r["Predicted risk"].astype(str).str.lower().eq("high").sum())); c.metric("Medium",int(r["Predicted risk"].astype(str).str.lower().eq("medium").sum())); d.metric("Low",int(r["Predicted risk"].astype(str).str.lower().eq("low").sum()))
        choices=st.multiselect("Filter risk",sorted(r["Predicted risk"].unique()),default=sorted(r["Predicted risk"].unique()))
        st.dataframe(r[r["Predicted risk"].isin(choices)],use_container_width=True,hide_index=True)
        st.markdown("### Explainable risk review")
        for _,row in r.iterrows():
            with st.expander(f"Clause {row['Clause #']} · {row['Predicted risk']} · {row['Predicted clause type']}"):
                st.write(row["Clause text"]); st.write("**Review reason:**",row["Risk explanation"]); st.write(f"Confidence: {row['Risk confidence']:.0%} (not proof of correctness)")
                st.info("Suggested action: compare this wording with company policy and ask a qualified reviewer.")
        st.download_button("Download analysis CSV",r.to_csv(index=False).encode(),"contractiq_analysis.csv","text/csv")

elif page=="AI assistant":
    st.markdown("### 💬 ContractIQ assistant")
    st.write("Ask about the current contract or request general review suggestions.")
    st.caption("This demo uses local TF-IDF retrieval and guided responses, not a connected generative AI model.")
    for q in ["What can you help me with?","Find payment terms","What should I review about liability?","Find termination conditions"]:
        if st.button(q): st.session_state.question=q
    with st.form("chat"):
        question=st.text_input("Your message",value=st.session_state.get("question",""),placeholder="Ask about payment, renewal, liability, privacy…")
        send=st.form_submit_button("Send",type="primary")
    if send and question.strip():
        st.session_state.chat.append(("You",question))
        q=clean(question)
        if any(w in q for w in ["hello","hi","hey"]):
            answer="Hello! I can help retrieve contract passages, explain model flags, run policy checks and suggest questions for your legal or procurement team."
        elif not st.session_state.clauses:
            answer="Please analyze a contract first so I can search its wording. I can also offer general review guidance, but contract-specific answers need the document."
        elif any(w in q for w in ["help","what can you","suggest"]):
            answer="I can retrieve relevant clauses, explain potential risk flags, and suggest what to verify. Try asking about payment, liability, renewal, privacy, or termination. I cannot replace qualified legal review."
        else:
            hits=retrieve(question,st.session_state.clauses)
            answer="I couldn't find a close match in the current contract. Try specific terms from the agreement." if not hits else "Here are relevant passages I found:\n\n" + "\n\n".join([f"**Passage {i+1} · similarity {score:.2f}**\n\n> {clause}" for i,score,clause in hits]) + "\n\n**Suggestion:** verify definitions, exceptions and surrounding clauses before making a decision."
        st.session_state.chat.append(("ContractIQ assistant",answer)); st.session_state.question=""
    for who,msg in st.session_state.chat:
        st.markdown(f"**{'🧑' if who=='You' else '🤖'} {who}**")
        st.markdown(msg)
        st.divider()
    if st.session_state.chat and st.button("Clear conversation"): st.session_state.chat=[]; st.rerun()

elif page=="Policy compliance":
    st.markdown("### 🛡️ Company policy checker")
    st.write("Set example review rules. These transparent heuristics can flag potential issues but cannot certify legal compliance.")
    a,b=st.columns(2)
    with a:
        cap=st.checkbox("Require a clear liability cap",True); days=st.number_input("Maximum payment period (days)",0,365,45); renew=st.checkbox("Flag automatic renewal",True)
    with b:
        privacy=st.checkbox("Review data privacy safeguards",True); term=st.checkbox("Review termination notice",True)
    if not st.session_state.clauses: st.info("Analyze a contract first.")
    elif st.button("Run policy review",type="primary"):
        findings=[]
        for i,c in enumerate(st.session_state.clauses,1):
            for sev,rule,detail,action in policy_findings(c,{"cap":cap,"days":int(days),"renew":renew,"privacy":privacy,"term":term}):
                findings.append({"Clause #":i,"Severity":sev,"Rule":rule,"Finding":detail,"Suggested action":action,"Clause text":c})
        if findings:
            out=pd.DataFrame(findings); st.warning(f"{len(out)} potential issue(s) detected."); st.dataframe(out,use_container_width=True,hide_index=True)
            st.download_button("Download compliance CSV",out.to_csv(index=False).encode(),"contractiq_compliance.csv","text/csv")
        else: st.success("No configured rule was triggered. This does not prove full compliance.")

elif page=="Export report":
    st.markdown("### 📥 Review report")
    r=st.session_state.results
    if r.empty: st.info("Analyze a contract first.")
    else:
        st.write("**Document:**",st.session_state.filename); st.dataframe(r,use_container_width=True,hide_index=True)
        st.download_button("Download complete CSV report",r.to_csv(index=False).encode(),"contractiq_report.csv","text/csv")

elif page=="About & security":
    st.markdown("### About this prototype")
    st.write(f"Training rows: {len(df)}. Models: TF-IDF + Logistic Regression for clause type and risk level.")
    st.warning("The login screen is a demo gate, not real authentication. Do not use real company credentials. Before real use, integrate approved SSO/OAuth, server-side authorization, secure storage, audit logs, and retention/deletion controls.")
    st.warning("The chatbot retrieves passages using TF-IDF; it is not a generative LLM. Compliance rules are heuristics. The small training dataset may produce unreliable predictions. Human review is required.")
    st.write("Never upload confidential contracts to public hosting unless approved by your organization.")

st.divider()
st.caption("ContractIQ • Enterprise portal prototype • AI outputs are decision support, not legal advice.")
