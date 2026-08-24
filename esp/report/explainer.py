"""The project explained to somebody who knows nothing about any of this.

The primer assumes a reader who knows what an AI agent is. This one assumes
nothing at all -- not agents, not models, not tokens, not evolution. Every idea
arrives attached to something the reader has already seen in ordinary life: a
restaurant kitchen, a wind tunnel, a hiring process, an electricity bill.

It makes exactly the same admissions as every other document here. A beginner's
version that quietly drops what did not work is not a simpler document, it is a
less true one, and the reader least able to check is the one who deserves it
most.
"""

from __future__ import annotations

from pathlib import Path

from reportlab.platypus import PageBreak, Spacer

from esp.report.layout import (
    ACCENT,
    AMBER,
    CW,
    GOOD,
    GOOD_BG,
    LEAD,
    SOFT,
    WARN_BG,
    Layout,
    _s,
)

ROOT = Path(__file__).resolve().parent.parent.parent

BIG = _s("big", fontSize=13, leading=18.5, spaceAfter=10)
STEP = _s("step", fontSize=10, leading=14.4, leftIndent=14, spaceAfter=7)
MUTED = "#5b6672"


class Explainer(Layout):
    def __init__(self, results_dir: str | Path | None = None):
        super().__init__(results_dir or ROOT / "results")

    # ---------------------------------------------------------------- helpers

    def life(self, title: str, text: str) -> None:
        """A real-life anchor. Every abstract idea in this document gets one."""
        self.callout(f"In real life &mdash; {title}", text, bg=SOFT, bar=ACCENT)

    def step(self, letter: str, title: str, text: str) -> None:
        self.p(f'<font color="#1a4fa0"><b>{letter}. {title}</b></font>',
               _s("st", fontSize=11, leading=14.6, spaceBefore=10, spaceAfter=3))
        self.p(text, STEP)

    # ---------------------------------------------------------------- content

    def build(self, out: str | Path | None = None) -> Path:
        self._a_what_is_an_agent()
        self._b_the_problem()
        self._c_the_analogy()
        self._d_how()
        self._e_the_result()
        self._f_forever()
        self._g_online()
        self._h_not_done()
        self._i_words()

        out = Path(out) if out else ROOT / "docs" / "neuro-san-esp-Explainer.pdf"
        out.parent.mkdir(parents=True, exist_ok=True)
        return self.render(out, "neuro-san-esp explained from scratch")

    # ------------------------------------------------------------------ A

    def _a_what_is_an_agent(self) -> None:
        self.h1("A. What is an AI agent, really?",
                "Starting from nothing. No prior knowledge assumed.")

        self.p(
            "An <b>AI agent</b> is a small program with a job description written in "
            "plain English, powered by an AI model. You do not program it with rules. "
            "You tell it what its job is, what tools it can use, and let it work out "
            "the rest.", LEAD)

        self.life(
            "a new employee on their first day",
            "You do not hand a new hire ten thousand lines of instructions. You say: "
            "<i>&ldquo;You handle customer refunds. Here is the refund system, here is "
            "the order database. Ask Priya in finance if the amount is over "
            "&pound;500.&rdquo;</i><br/><br/>"
            "That is an agent. A job description, some tools, and permission to ask a "
            "colleague. The AI model is the part that reads the situation and decides "
            "what to do next.")

        self.h2("And an agent network?")
        self.p(
            "One agent doing everything is like one person running a whole restaurant "
            "&mdash; possible, slow, and they will drop something. So you use several, "
            "each with a narrower job, and one of them takes the order and decides who "
            "to pass it to.")

        self.life(
            "a restaurant kitchen",
            "A customer asks for a dish. The <b>head chef</b> does not cook it alone "
            "&mdash; they pass the fish to the fish station, the sauce to the sauce "
            "cook, the dessert to pastry, then plate it up and send it out.<br/><br/>"
            "The customer only ever speaks to one person. Behind that, four "
            "specialists did the work. <b>That is an agent network</b>, and the head "
            "chef is what this project calls the &ldquo;front man&rdquo;.")

        self.story.append(PageBreak())

    # ------------------------------------------------------------------ B

    def _b_the_problem(self) -> None:
        self.h1("B. The problem this project solves",
                "One sentence, and it is the whole reason the project exists.")

        self.p(
            "Cognizant AI Lab publishes a free framework called <b>neuro-san</b> that "
            "builds these agent networks for you. Describe the job in one sentence and "
            "it designs a whole team of agents in about five seconds. It is genuinely "
            "impressive.", LEAD)

        self.callout(
            "It designs one team. It never checks whether the team is any good.",
            "There is <b>no scoring function anywhere in the framework</b>. So nobody "
            "can answer the obvious questions: is a nine-agent team better than a "
            "five-agent one for this job? Should every agent use the expensive AI "
            "model, or would a cheap one do for most of them? Are the job "
            "descriptions it wrote actually clear?<br/><br/>"
            "It is design with no measurement.",
            bg=WARN_BG, bar=AMBER)

        self.life(
            "an architect who never checks the building stands up",
            "Imagine an architect who can sketch a house in five seconds. Brilliant. "
            "Now imagine they never calculate whether the roof holds, never price the "
            "materials, and never compare their design to any other.<br/><br/>"
            "You would not call that finished. You would say the <i>drawing</i> is "
            "done and the <i>engineering</i> has not started. That is exactly the gap "
            "here &mdash; and this project is the engineering half.")

        self.h2("Why it matters in money")
        self.p(
            "Two agent teams can give you <b>the same answers</b> and send you very "
            "different bills. AI providers charge by the amount of text processed, so "
            "a badly-shaped team quietly costs more forever, and nothing tells you.")

        self.life(
            "two electricians, same job",
            "Both rewire your house. Both pass inspection. One takes four hours, the "
            "other takes seven and charges you for it.<br/><br/>"
            "Identical result, very different invoice. You would want to know that "
            "<i>before</i> hiring, not after. Right now, with AI agent teams, there is "
            "no way to know.")

        self.story.append(PageBreak())

    # ------------------------------------------------------------------ C

    def _c_the_analogy(self) -> None:
        self.h1("C. The clever bit, in one picture",
                "Why the method is called ESP, and why it is affordable.")

        self.p(
            "Testing an agent team for real is slow and costs money &mdash; about "
            "<b>eight minutes</b> and a real bill each time. Trying thousands of ideas "
            "that way is impossible.", BIG)

        self.life(
            "designing a car in a wind tunnel",
            "Real wind-tunnel tests are slow and expensive, so car engineers do "
            "this:<br/><br/>"
            "<b>1.</b> Test a handful of real shapes in the tunnel. Slow, costly.<br/>"
            "<b>2.</b> Use those results to build a cheap computer simulator that "
            "<i>guesses</i> how a shape will perform.<br/>"
            "<b>3.</b> Try ten thousand shapes in the simulator overnight. Free, "
            "instant.<br/>"
            "<b>4.</b> Put only the best few back in the real tunnel.<br/>"
            "<b>5.</b> Feed those new real results back into the simulator, so its "
            "guesses get better. Repeat.")

        self.p(
            "That is precisely what this project does, with agent teams instead of car "
            "bodies. The method has a name &mdash; <b>ESP</b>, Evolutionary "
            "Surrogate-assisted Prescription &mdash; and it was invented by Cognizant "
            "AI Lab, the same people who wrote neuro-san. The &ldquo;surrogate&rdquo; "
            "is the simulator.")

        self.h2("The numbers behind the picture, measured here")
        self.table(
            ["", "Real test (the wind tunnel)", "The simulator's guess"],
            [["One team", "about 8 minutes, real money", "<b>a fraction of a second</b>"],
             ["2,000 teams", "roughly <b>267 hours</b>", "<b>about 0.14 seconds</b>"],
             ["Cost", "a real bill each time", "<b>nothing at all</b>"]],
            widths=[CW * 0.26, CW * 0.37, CW * 0.37])

        self.callout(
            "This is the whole trick",
            "Trying two thousand designs for real would take eleven days and a large "
            "bill. Trying two thousand <i>guesses</i> takes a tenth of a second and "
            "costs nothing. You then pay for only the handful the guess says are worth "
            "it.<br/><br/>"
            "The guessing is free. That is what makes searching affordable at all.",
            bg=GOOD_BG, bar=GOOD)

        self.story.append(PageBreak())

    # ------------------------------------------------------------------ D

    def _d_how(self) -> None:
        self.h1("D. How it works, step by step",
                "Six steps. No jargon.")

        self.step("1", "Invent a company that does not exist",
            "A fictional logistics firm: 24 depots, 40 contracts, 60 incident reports, "
            "124 documents. All made up, on purpose. If we used a real company, the AI "
            "might already know the answers from its training and would look clever "
            "without actually reading anything. Nothing about this company exists "
            "anywhere, so the only way to answer is to go and look it up.")

        self.life(
            "an exam with a made-up textbook",
            "If you set an exam on famous history, a student can bluff from general "
            "knowledge. If you set it on a novel written last week that nobody has "
            "read, the only way to pass is to actually read the book. That is why the "
            "company is invented.")

        self.step("2", "Write 17 questions where you already know the answer",
            "They range from easy &mdash; <i>&ldquo;who manages depot D08?&rdquo;</i> "
            "&mdash; to hard, needing four documents joined together: <i>&ldquo;incident "
            "INC-4413 hit a contract, that contract is served by a depot, who manages "
            "that depot?&rdquo;</i> Because the company and the questions come from the "
            "same generator, every answer is correct by construction. Nobody "
            "hand-wrote an answer that might be wrong.")

        self.step("3", "Score a team by actually running it",
            "Give the team all 17 questions and count three things: how many it got "
            "right, how much text it burned through (the bill), and how many agents it "
            "used. Getting the answer right matters most &mdash; a cheap team that "
            "answers nothing is worthless &mdash; but cost and size count too.")

        self.step("4", "Change one thing and see what happens",
            "Seven kinds of change: add an agent, remove one, change who reports to "
            "whom, split one agent into two, merge two into one, change who is allowed "
            "to search the documents, and swap an agent onto a cheaper or more "
            "expensive AI model. Each change is checked for sanity first, and a broken "
            "one is <b>thrown away rather than patched up</b>.")

        self.life(
            "reorganising a team at work",
            "Hire someone. Let someone go. Change who reports to whom. Split one "
            "overloaded role into two. Merge two quiet ones. Give someone access to "
            "the shared drive. Move someone from a contractor rate to a staff rate.<br/>"
            "<br/>Those are exactly the seven changes &mdash; and just like at work, "
            "most of them make things slightly worse and a few make things much better. "
            "You only find out by trying.")

        self.step("5", "Learn to guess the score without running anything",
            "Look at the <i>shape</i> of a team &mdash; how many agents, how deep the "
            "chain of command, who talks to whom, which model each one uses &mdash; and "
            "predict how well it will score. This is the simulator from the wind tunnel. "
            "It learns from the real results collected so far, so it starts out poor and "
            "improves as more real results arrive.")

        self.step("6", "Try thousands, pay for a few, repeat forever",
            "Generate thousands of changed teams, guess all their scores for free, keep "
            "the best few, and pay to really run only those. Their real results go back "
            "into training the guesser, which then guesses better next time.")

        self.story.append(PageBreak())

    # ------------------------------------------------------------------ E

    def _e_the_result(self) -> None:
        self.h1("E. What it actually found",
                "The finding that makes the whole thing worth doing.")

        self.p(
            "Three different team designs were run for real against all 17 questions. "
            "The important column is the last one.", BIG)

        self.table(
            ["The design", "Questions right", "Cost (text processed)", "Agents"],
            [["One agent doing everything", "high", "<b>the most expensive</b>", "1"],
             ["A flat pair of agents", "high", "middle", "3"],
             ["The shape neuro-san's designer produces", "high", "<b>the cheapest</b>",
              "4"]],
            widths=[CW * 0.42, CW * 0.18, CW * 0.26, CW * 0.14], highlight=[2])

        self.callout(
            "They answered about equally well. The bills were nowhere near equal.",
            "The cheapest design costs roughly <b>half</b> what the dearest one does, "
            "for the same work on the same questions with the same AI model.<br/><br/>"
            "The <i>shape</i> of the team barely changed what it could do. It changed "
            "what doing it <b>cost</b>. On a real system answering thousands of "
            "questions a day, that is an enormous bill for nothing &mdash; and today "
            "there is no way to see it, because the framework cannot measure any of "
            "this.")

        self.life(
            "your electricity bill",
            "Two houses, same size, same family, same comfort. One pays double. Not "
            "because anyone is doing anything differently &mdash; because of insulation "
            "and boiler choices nobody ever measured.<br/><br/>"
            "Nobody thinks about it until somebody finally measures it. This project is "
            "the meter.")

        self.h2("A warning about precise numbers")
        self.p(
            "Earlier versions of this document quoted exact figures and an exact "
            "percentage. Those specific numbers turned out to be partly an <b>artifact "
            "of a measurement bug</b> &mdash; questions that timed out were being "
            "counted as wrong answers, which made three different outcomes look "
            "identical. After the fix, the scores moved.<br/><br/>"
            "What survived the fix is the part that matters: <b>the cost ordering is "
            "stable, and the gap is large</b>. The cheapest design really is far "
            "cheaper. But quote the ordering, not a decimal.",
            _s("warn2", fontSize=9.6, leading=13.6, textColor=MUTED))

        self.story.append(PageBreak())

    # ------------------------------------------------------------------ F

    def _f_forever(self) -> None:
        self.h1("F. Why it runs by itself, forever",
                "The part that makes it a product rather than a school project.")

        self.p(
            "The AI provider gives away a limited amount free: about <b>500 requests a "
            "day</b>. Testing one team costs about 165 of them. So a whole day buys "
            "<b>three tests</b>.", BIG)

        self.p(
            "The first version of this project was a script that planned forty tests "
            "and ran them all at once. It failed every single time &mdash; not from a "
            "bug, but from arithmetic. Forty tests need about thirteen days of "
            "allowance.")

        self.callout(
            "The change of mind that fixed it",
            "The daily limit is not an obstacle to a service. It is its "
            "<b>rhythm</b>.<br/><br/>"
            "Instead of one big run that dies, it now wakes up every hour, spends "
            "whatever allowance is left, writes down what it learned, and stops. "
            "Tomorrow it carries on. Over weeks it builds up a collection of results "
            "that could never be bought in one afternoon &mdash; and nobody has to be "
            "watching.",
            bg=GOOD_BG, bar=GOOD)

        self.life(
            "saving up rather than borrowing",
            "You cannot buy the thing this month. You can put aside what is spare each "
            "month and have it by spring. Nothing clever &mdash; but it needs a system "
            "that survives you forgetting about it, and a record that survives being "
            "interrupted halfway.<br/><br/>"
            "That is why it saves after <i>every single test</i> rather than at the end. "
            "Being interrupted is the expected ending, not a failure.")

        self.story.append(PageBreak())

    # ------------------------------------------------------------------ G

    def _g_online(self) -> None:
        self.h1("G. You can talk to it",
                "The winner is a thing you can use, not a number in a table.")

        self.p(
            "A search that finishes with a winning identifier in a spreadsheet has "
            "stopped one step short. The reason to measure teams is that one of them is "
            "better to <b>actually use</b>.", LEAD)

        self.p(
            "One command puts the best-measured team behind a web page. You type a "
            "question, the real agents run, and if your question happens to be one of "
            "the 17 graded ones, it is <b>marked against the known answer in front of "
            "you</b> &mdash; right or wrong, no hiding.")

        placed = self.image(
            ROOT / "docs" / "screenshots" / "web-02-answered.png",
            "The page, answering. Four agents, and the answer graded correct against "
            "the known ground truth.", width=CW * 0.84)
        if not placed:
            self.p("(screenshot not captured in this build)",
                   _s("miss", fontSize=9, textColor=MUTED))

        self.callout(
            "A four-hop question, answered correctly, live",
            "<font face='Courier' size='9'>Incident INC-4413 affected a contract "
            "serviced by a depot. Who is that depot&rsquo;s manager?</font><br/><br/>"
            "The team had to find the incident, follow it to a contract, follow that to "
            "a depot, then look up who runs it &mdash; four documents, chained. It "
            "answered <b>R. Delacroix</b>, which is correct, in about 48 seconds.",
            bg=GOOD_BG, bar=GOOD)

        self.story.append(PageBreak())

    # ------------------------------------------------------------------ H

    def _h_not_done(self) -> None:
        self.h1("H. What it has not done",
                "The part most documents leave out.")

        self.callout(
            "The search has not yet found a better team than the starting one",
            "Only a handful of real tests exist. Finding a better design needs a "
            "collection of results, a collection needs allowance, and allowance arrives "
            "at three tests a day. That is the honest state of it. If it never beats "
            "the starting design, that will be reported as the answer.",
            bg=WARN_BG, bar=AMBER)

        self.callout(
            "The guesser has never actually been trained",
            "It needs eight real results before it can learn anything; there are fewer. "
            "Below that it returns the same guess for every design, and the code now "
            "says so instead of printing a confident-looking number.<br/><br/>"
            "An earlier version printed <font face='Courier' size='9'>+0.000</font> in "
            "that case, which looked exactly like a measured result of zero. It was a "
            "placeholder. That was quoted as a real measurement for a while &mdash; "
            "including by me &mdash; and it was wrong.",
            bg=WARN_BG, bar=AMBER)

        self.h2("Things learned only by running it")
        self.p(
            "Roughly a dozen bugs were found this way, and nearly all of them made a "
            "<b>good</b> design look bad. That is the dangerous kind, because a search "
            "being fed wrong information still produces a smooth, convincing-looking "
            "graph.", _s("intro2", fontSize=9.8, leading=13.6, spaceAfter=8))

        for title, text in [
            ("The wrong agent was answering the question",
             "The framework treats whichever agent is listed <i>first</i> as the one "
             "who receives the question. Listing them alphabetically meant one design "
             "was fronted by an arithmetic agent, which replied &ldquo;you did not "
             "provide any numbers&rdquo; to all 17 questions and scored zero. It had "
             "never actually been run as designed &mdash; and neither had any other "
             "measurement taken before that."),
            ("Timeouts were being counted as wrong answers",
             "A question that ran out of time looked identical to a question answered "
             "incorrectly. That made three quite different designs appear to score "
             "exactly the same, which became a headline finding. It was an artifact."),
            ("A test certified a bug instead of catching it",
             "A function was supposed to use world time and used local time instead. "
             "The test that was meant to check this asserted local time too &mdash; so "
             "it passed, permanently, while being wrong."),
            ("A missing ingredient nobody could notice locally",
             "A required package was never declared. It worked forever on the machine "
             "that happened to have it already. Only a completely fresh install could "
             "reveal it."),
        ]:
            self.callout(title, text)

        self.life(
            "the smoke alarm you never tested",
            "It looks fine. It has a light on it. You would only discover it does "
            "nothing on the one night it matters.<br/><br/>"
            "Most of these bugs were that: something that looked like it was working, "
            "reporting confident numbers, and quietly wrong. Which is the argument for "
            "measuring in the first place.")

        self.story.append(PageBreak())

    # ------------------------------------------------------------------ I

    def _i_words(self) -> None:
        self.h1("I. Every word, in plain English",
                "In case any of them were new.")

        self.table(
            ["Word", "What it means"],
            [["AI model",
              "The engine that reads text and writes text. You rent it by the word."],
             ["Agent",
              "A small program with a plain-English job description, powered by a "
              "model. Like a new hire with a role and some tools."],
             ["Agent network",
              "Several agents wired together, one of whom takes the question and asks "
              "the others. Like a restaurant kitchen."],
             ["Topology",
              "The <i>shape</i> of that team: how many, and who talks to whom."],
             ["neuro-san",
              "Cognizant AI Lab's free framework for building agent networks. This "
              "project sits on top of it."],
             ["ESP",
              "Evolutionary Surrogate-assisted Prescription. Cognizant AI Lab's "
              "method: build a cheap guesser, search against it for free, pay only for "
              "the best few. The wind tunnel."],
             ["Surrogate / guesser",
              "The cheap stand-in that estimates a score without running anything."],
             ["Fitness",
              "The score a design gets: right answers, cost, and team size combined."],
             ["Token",
              "Roughly a word. Providers bill by these, so &ldquo;tokens&rdquo; and "
              "&ldquo;cost&rdquo; mean the same thing here."],
             ["Quota / rate limit",
              "The provider's cap on free use. Here: about 500 requests a day."],
             ["Baseline",
              "The design you are trying to beat &mdash; the one neuro-san's own "
              "designer would have handed you."]],
            widths=[CW * 0.20, CW * 0.80])

        self.story.append(Spacer(1, 10))
        self.callout(
            "The whole project in one sentence",
            "neuro-san can design a team of AI agents but cannot tell you whether the "
            "team is any good; this measures that, searches for a better team, keeps "
            "searching by itself, and lets you talk to the winner &mdash; and the first "
            "thing it measured was that two teams doing the same job can cost twice as "
            "much as each other.",
            bg=SOFT, bar=ACCENT)

        self.p(
            "Code and full technical write-up: "
            "<font color='#1a4fa0'>github.com/Sivakumarraj/neuro-san-esp</font>",
            _s("end", fontSize=9.6, leading=13, textColor=MUTED, spaceBefore=8))


if __name__ == "__main__":
    print("wrote", Explainer().build())
