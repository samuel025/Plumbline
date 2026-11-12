"""Two fictional churches (synthetic data only) with short statements of faith."""
from . import db

CHURCHES = [
 ("Harbor Grace Church (fictional)", "Reformed Baptist", "Warm, plain, Scripture-saturated.", [
  ("Scripture is the inspired, inerrant and sufficient Word of God.", "scripture"),
  ("God is one in essence and three in persons: Father, Son and Holy Spirit.", "trinity"),
  ("Salvation is by grace alone, through faith alone, in Christ alone; works are the fruit, not the root.", "salvation"),
  ("God sovereignly elects people to salvation before the foundation of the world.", "election"),
  ("Baptism is for believers only, by immersion, as a sign of faith already professed.", "baptism"),
  ("Christ is spiritually present in the Lord's Supper, which is a memorial and not a sacrifice.", "communion"),
  ("The miraculous sign gifts ceased with the apostolic age.", "spiritual gifts"),
  ("Only men may serve as pastors and elders.", "church order"),
  ("Christ will return personally and visibly to judge the living and the dead.", "eschatology"),
  ("Human life is sacred from conception to natural death.", "ethics"),
  ("Believers suffer in a fallen world and are never promised health or wealth in this life.", "suffering"),
  ("Serious pastoral crises are handled by pastors, not by automated systems.", "pastoral care"),
 ]),
 ("Meridian Fellowship (fictional)", "Charismatic non-denominational", "Encouraging, expectant, conversational.", [
  ("Scripture is the inspired Word of God and our final authority.", "scripture"),
  ("God is one in essence and three in persons: Father, Son and Holy Spirit.", "trinity"),
  ("Salvation is by grace through faith; every person is free to accept or refuse it.", "salvation"),
  ("Christ calls all people to himself; God desires everyone to be saved.", "election"),
  ("Baptism is for believers; infants are dedicated, not baptized.", "baptism"),
  ("The gifts of the Spirit, including healing and prophecy, are active today.", "spiritual gifts"),
  ("God may heal today, and the church prays for healing, while honoring medical care.", "healing"),
  ("Women and men may both serve as pastors and leaders.", "church order"),
  ("Christ will return personally and visibly.", "eschatology"),
  ("Human life is sacred from conception to natural death.", "ethics"),
  ("Believers should never replace professional medical or counseling help with prayer alone.", "pastoral care"),
  ("Serious pastoral crises are handled by pastors, not by automated systems.", "pastoral care"),
 ]),
]

def seed(force=False):
    db.init()
    with db.session() as c:
        if c.execute("SELECT COUNT(*) FROM church").fetchone()[0] and not force:
            return
        for name, trad, tone, comms in CHURCHES:
            cid = c.execute("INSERT INTO church(name,tradition,tone_guide) VALUES(?,?,?)", (name, trad, tone)).lastrowid
            for text, topic in comms:
                c.execute("INSERT INTO commitment(church_id,text,topic,status) VALUES(?,?,?,'approved')", (cid, text, topic))

if __name__ == "__main__":
    seed(); print("seeded")
