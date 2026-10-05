"""Raw industry text -> GICS Industry Name (exhaustive taxonomy).
Order: manual override on normalized key -> fuzzy match to override keys (OCR/typo noise) -> embedding nearest neighbour
over GICS sub-industry/industry names + override keys."""
import re, os, unicodedata, pandas as pd, numpy as np
from rapidfuzz import process, fuzz
from embed import embed
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = pd.read_csv(f'{ROOT}/task/input/schemamatching/GICS_Industry_Taxonomy.csv')
GICS = sorted(G['Industry Name'].unique())
O = {  # manual semantic judgements (key = normalized raw)
 'retail':'Broadline Retail','department stores':'Broadline Retail','discount stores':'Consumer Staples Distribution & Retail','stores':'Broadline Retail',
 'internet catalog retail':'Broadline Retail','e commerce':'Broadline Retail','online retail':'Broadline Retail','supermarket':'Consumer Staples Distribution & Retail',
 'food retail':'Consumer Staples Distribution & Retail','drug retail':'Consumer Staples Distribution & Retail','food drug retail':'Consumer Staples Distribution & Retail',
 'grocery':'Consumer Staples Distribution & Retail','supermarkets':'Consumer Staples Distribution & Retail','convenience store':'Consumer Staples Distribution & Retail',
 'specialty stores':'Specialty Retail','home improvement retail':'Specialty Retail','apparel footwear retail':'Specialty Retail','specialty retail':'Specialty Retail',
 'computer electronics retail':'Specialty Retail','automotive':'Automobiles','automotive industry':'Automobiles','auto industry':'Automobiles','automobile':'Automobiles',
 'automobiles':'Automobiles','auto truck manufacturers':'Automobiles','car':'Automobiles','motorcycle':'Automobiles','auto truck parts':'Automobile Components',
 'automotive parts':'Automobile Components','tire':'Automobile Components','tires':'Automobile Components',
 'telecommunication':'Diversified Telecommunication Services','telecommunications':'Diversified Telecommunication Services','telecom':'Diversified Telecommunication Services',
 'telecommunications services':'Diversified Telecommunication Services','telecom services':'Diversified Telecommunication Services',
 'telecommunication services':'Diversified Telecommunication Services','internet service provider':'Diversified Telecommunication Services',
 'mobile phone':'Wireless Telecommunication Services','mobile telecommunications':'Wireless Telecommunication Services','wireless':'Wireless Telecommunication Services',
 'video game':'Entertainment','video game industry':'Entertainment','video games':'Entertainment','game industry':'Entertainment','interactive entertainment':'Entertainment',
 'entertainment':'Entertainment','film':'Entertainment','film industry':'Entertainment','filmmaking':'Entertainment','film production':'Entertainment','music':'Entertainment',
 'music industry':'Entertainment','animation':'Entertainment','mixed martial arts':'Entertainment','professional wrestling':'Entertainment','pornography':'Entertainment',
 'record label':'Entertainment','motion picture':'Entertainment','television':'Media','software':'Software','software programming':'Software',
 'financial services':'Financial Services','finance services':'Financial Services','finance':'Financial Services','financial':'Financial Services',
 'thrifts mortgage finance':'Financial Services','diversified financial services':'Financial Services',
 'investment services':'Capital Markets','investment':'Capital Markets','venture capital':'Capital Markets','private equity':'Capital Markets',
 'investment management':'Capital Markets','asset management':'Capital Markets','investment banking':'Capital Markets','stock exchange':'Capital Markets',
 'consumer financial services':'Consumer Finance','consumer finance':'Consumer Finance','credit card':'Consumer Finance',
 'regional banks':'Banks','major banks':'Banks','banks':'Banks','bank':'Banks','banking':'Banks','regional':'Banks','banks regional':'Banks','large banks':'Banks',
 'retail banking':'Banks','commercial bank':'Banks','insurance':'Insurance','diversified insurance':'Insurance','life health insurance':'Insurance',
 'property casualty insurance':'Insurance','insurance brokers':'Insurance','health insurance':'Insurance','reinsurance':'Insurance',
 'oil gas operations':'Oil Gas & Consumable Fuels','oil and gas operations':'Oil Gas & Consumable Fuels','petroleum industry':'Oil Gas & Consumable Fuels',
 'petroleum':'Oil Gas & Consumable Fuels','oil and gas':'Oil Gas & Consumable Fuels','oil gas':'Oil Gas & Consumable Fuels','natural gas':'Oil Gas & Consumable Fuels',
 'coal':'Oil Gas & Consumable Fuels','coal mining':'Oil Gas & Consumable Fuels','energy':'Oil Gas & Consumable Fuels','energy industry':'Oil Gas & Consumable Fuels',
 'oil services equipment':'Energy Equipment & Services','oil gas equipment services':'Energy Equipment & Services',
 'electric utilities':'Electric Utilities','power utilities':'Electric Utilities','electricity':'Electric Utilities','electric utility':'Electric Utilities',
 'electric power industry':'Electric Utilities','electric power':'Electric Utilities','natural gas utilities':'Gas Utilities','diversified utilities':'Multi-Utilities',
 'public utility':'Multi-Utilities','utilities':'Multi-Utilities','water':'Water Utilities','water utilities':'Water Utilities',
 'electricity generation':'Independent Power and Renewable Electricity Producers','renewable energy':'Independent Power and Renewable Electricity Producers',
 'solar energy':'Independent Power and Renewable Electricity Producers','wind power':'Independent Power and Renewable Electricity Producers',
 'rail transport':'Ground Transportation','railroads':'Ground Transportation','transport':'Ground Transportation','public transport':'Ground Transportation',
 'other transportation':'Ground Transportation','trucking':'Ground Transportation','bus':'Ground Transportation','railway':'Ground Transportation','transportation':'Ground Transportation',
 'freight transport':'Air Freight & Logistics','logistics':'Air Freight & Logistics','air courier':'Air Freight & Logistics','mail':'Air Freight & Logistics',
 'courier':'Air Freight & Logistics','shipping':'Marine Transportation','marine transportation':'Marine Transportation',
 'airline':'Passenger Airlines','airlines':'Passenger Airlines','airport':'Transportation Infrastructure','real estate':'Real Estate Management & Development',
 'real estate development':'Real Estate Management & Development','property management':'Real Estate Management & Development',
 'aerospace':'Aerospace & Defense','aerospace defense':'Aerospace & Defense','aviation':'Aerospace & Defense','firearm':'Aerospace & Defense',
 'arms industry':'Aerospace & Defense','defense':'Aerospace & Defense','military':'Aerospace & Defense','defence':'Aerospace & Defense',
 'manufacturing':'Machinery','heavy equipment':'Machinery','other industrial equipment':'Machinery','shipbuilding':'Machinery','industrial machinery':'Machinery',
 'electronics':'Electronic Equipment Instruments & Components','electronic components':'Electronic Equipment Instruments & Components',
 'conglomerate company':'Industrial Conglomerates','conglomerate':'Industrial Conglomerates','conglomerates':'Industrial Conglomerates',
 'restaurant':'Hotels Restaurants & Leisure','restaurants':'Hotels Restaurants & Leisure','fast food':'Hotels Restaurants & Leisure','hotel':'Hotels Restaurants & Leisure',
 'hotels motels':'Hotels Restaurants & Leisure','hospitality':'Hotels Restaurants & Leisure','tourism':'Hotels Restaurants & Leisure','casinos gaming':'Hotels Restaurants & Leisure',
 'travel':'Hotels Restaurants & Leisure','gambling':'Hotels Restaurants & Leisure','casino':'Hotels Restaurants & Leisure','fashion':'Textiles Apparel & Luxury Goods',
 'clothing':'Textiles Apparel & Luxury Goods','watch':'Textiles Apparel & Luxury Goods','apparel accessories':'Textiles Apparel & Luxury Goods','textile':'Textiles Apparel & Luxury Goods',
 'footwear':'Textiles Apparel & Luxury Goods','luxury goods':'Textiles Apparel & Luxury Goods','jewellery':'Textiles Apparel & Luxury Goods','jewelry':'Textiles Apparel & Luxury Goods',
 'health care':'Health Care Providers & Services','healthcare services':'Health Care Providers & Services','healthcare':'Health Care Providers & Services',
 'managed health care':'Health Care Providers & Services','hospital':'Health Care Providers & Services','medical equipment supplies':'Health Care Equipment & Supplies',
 'medical devices':'Health Care Equipment & Supplies','medical device':'Health Care Equipment & Supplies',
 'publishing':'Media','mass media':'Media','broadcasting':'Media','broadcasting cable':'Media','printing publishing':'Media','advertising':'Media','marketing':'Media',
 'newspaper':'Media','news media':'Media','media':'Media','radio':'Media','cable television':'Media','book publishing':'Media',
 'food':'Food Products','food processing':'Food Products','food industry':'Food Products','agriculture':'Food Products','confectionery':'Food Products','dairy':'Food Products',
 'meat':'Food Products','processing':'Food Products','alcoholic beverage':'Beverages','beverages':'Beverages','beverage':'Beverages','drink':'Beverages','soft drink':'Beverages',
 'brewing':'Beverages','brewery':'Beverages','beer':'Beverages','wine':'Beverages','winery':'Beverages','distillery':'Beverages','tobacco':'Tobacco',
 'construction services':'Construction & Engineering','construction':'Construction & Engineering','engineering':'Construction & Engineering',
 'general contractor':'Construction & Engineering','construction materials':'Construction Materials','cement':'Construction Materials',
 'iron steel':'Metals & Mining','steel':'Metals & Mining','metal':'Metals & Mining','mining':'Metals & Mining','diversified metals mining':'Metals & Mining','aluminum':'Metals & Mining',
 'metals mining':'Metals & Mining','precious metals':'Metals & Mining','gold mining':'Metals & Mining',
 'internet':'Interactive Media & Services','search engine':'Interactive Media & Services','social media':'Interactive Media & Services','social networking service':'Interactive Media & Services',
 'information technology':'IT Services','computer services':'IT Services','it services':'IT Services','information technology consulting':'IT Services',
 'it service management':'IT Services','computer network':'IT Services','consulting':'Professional Services','management consulting':'Professional Services',
 'business personal services':'Professional Services','staffing':'Professional Services','business services':'Professional Services',
 'computer hardware':'Technology Hardware Storage & Peripherals','computer storage devices':'Technology Hardware Storage & Peripherals','computers':'Technology Hardware Storage & Peripherals',
 'semiconductor':'Semiconductors & Semiconductor Equipment','semiconductors':'Semiconductors & Semiconductor Equipment','semiconductor industry':'Semiconductors & Semiconductor Equipment',
 'communications equipment':'Communications Equipment','telecommunications equipment':'Communications Equipment','networking equipment':'Communications Equipment',
 'consumer electronics':'Household Durables','household appliances':'Household Durables','home appliance':'Household Durables','pottery':'Household Durables',
 'furniture':'Household Durables','homebuilding':'Household Durables','household personal care':'Household Products','household products':'Household Products',
 'personal care':'Personal Care Products','cosmetics':'Personal Care Products','biotechnology':'Biotechnology','biotechs':'Biotechnology',
 'pharmaceuticals':'Pharmaceuticals','pharma':'Pharmaceuticals','pharmaceutical drug':'Pharmaceuticals','pharmaceutical':'Pharmaceuticals','pharmaceutical industry':'Pharmaceuticals',
 'chemical industry':'Chemicals','chemicals':'Chemicals','chemical substance':'Chemicals','diversified chemicals':'Chemicals','specialized chemicals':'Chemicals','chemical':'Chemicals',
 'trading companies':'Trading Companies & Distributors','trading':'Trading Companies & Distributors','rental leasing':'Trading Companies & Distributors',
 'wholesale':'Trading Companies & Distributors','distribution':'Distributors','bicycle':'Leisure Products','recreational products':'Leisure Products','toy':'Leisure Products','toys':'Leisure Products',
 'sporting goods':'Leisure Products','stationery':'Commercial Services & Supplies','environmental waste':'Commercial Services & Supplies','security systems':'Commercial Services & Supplies',
 'waste management':'Commercial Services & Supplies','printing':'Commercial Services & Supplies','education':'Diversified Consumer Services',
 'paper paper products':'Paper & Forest Products','paper products':'Paper & Forest Products','paper':'Paper & Forest Products','forest products':'Paper & Forest Products',
 'containers packaging':'Containers & Packaging','packaging':'Containers & Packaging','electrical equipment':'Electrical Equipment',
 'reit':'Diversified REITs','technology':'IT Services','chocolate':'Food Products','paint':'Chemicals','computer security':'Software',
 'internet security':'Software','network security':'Software','information security':'Software','business software':'Software','biofuel':'Oil Gas & Consumable Fuels',
 'collectable':'Leisure Products','architecture':'Professional Services','funeral director':'Diversified Consumer Services','physical fitness':'Hotels Restaurants & Leisure',
 'computer aided design':'Software','software as a service':'Software','cloud computing':'IT Services','stock photography':'Media','3d printing':'Machinery','real estate investment trust':'Diversified REITs',
}
def nkey(s):
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii','ignore').decode().lower()
    s = s.translate(str.maketrans({'0':'o','1':'l','3':'e','5':'s','8':'b'}))
    s = re.sub(r'\(.*?\)', lambda m: m.group(0)[1:-1], s)
    s = re.sub(r'[^a-z ]', ' ', s)
    toks = [t for t in s.split() if t not in ('and','the','of','industry','company')]
    return ' '.join(toks)
OKEYS = {}
for k, v in O.items():
    OKEYS[' '.join(sorted(nkey(k).split()))] = v
_cache = {}
def _refs():
    refs = [(r['Sub-Industry Name'], r['Industry Name']) for _, r in G.iterrows()] + [(n, n) for n in GICS] + [(k, v) for k, v in O.items()]
    return refs
def map_all(raws):
    raws = sorted({r for r in raws if isinstance(r, str) and r.strip()})
    out = {}; need = []
    for r in raws:
        k = nkey(r); sk = ' '.join(sorted(k.split()))
        if not k: out[r] = (None, 'empty'); continue
        if r.strip() in GICS: out[r] = (r.strip(), 'exact'); continue
        if sk in OKEYS: out[r] = (OKEYS[sk], 'override'); continue
        m = process.extractOne(sk, list(OKEYS), scorer=fuzz.ratio)
        if m and m[1] >= 88: out[r] = (OKEYS[m[0]], f'override_fuzzy{int(m[1])}'); continue
        need.append(r)
    if need:
        refs = _refs()
        R = embed([a for a, _ in refs]); V = embed(need)
        S = V @ R.T
        for r, row in zip(need, S):
            i = int(row.argmax()); out[r] = (refs[i][1], f'embed{row[i]:.2f}')
    return out
