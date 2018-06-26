import logging
logger = logging.getLogger(__name__)
import regex as re
import pandas as pd
import pyltp

ILLEGAL_CHARACTERS_RE = re.compile(r'[\000-\010]|[\013-\014]|[\016-\037]|[\x00-\x1f\x7f-\x9f]|[\uffff]')


def illegal_char_remover(data):
    if isinstance(data, str):
        return ILLEGAL_CHARACTERS_RE.sub(r'', data)
    else:
        return data


def re_sub(text):
    if isinstance(text, str) and (text is not None):
        text_s = re.sub('#.*?#|\[.*?][:：]*|【.*?】[:：]*', '', text)
        text_s = re.sub('https?:[a-zA-Z\\/\\.0-9_]+', '', text_s)
        text_s = re.sub('@.+?[,，：:\ )]+|@.+?$', '', text_s)
        text_s = re.sub('我在(\\w){0,2}[:：](\\w*)', '', text_s)
        text_s = re.sub('\\[(\\w){1,4}\\]', '', text_s)
        text_s = re.sub('\s+', ' ', text_s)
        text_s = re.sub('<b>|</b>|<br/>', '', text_s)
        text_s = re.sub(',', '，', text_s)
        text_s = re.sub('，+', '，', text_s)
        text_s = re.sub('❓', '？', text_s)
        text_s = re.sub('[❗‼！]+', '！', text_s)
        text_s = re.sub('~+', '。', text_s)
        text_s = re.sub('[.。]+', '。', text_s)
        text_s = re.sub(r'([\u4e00-\u9fa5A-Za-z0-9]) ([\u4e00-\u9fa5A-Za-z0-9])', r'\1，\2', text_s)
        try:
            # python UCS-4 build的处理方式
            highpoints = re.compile(u'[\U00010000-\U0010ffff]')
        except re.error:
            # python UCS-2 build的处理方式
            highpoints = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')
        text_s = highpoints.sub(u'', text_s)
        ta = re.compile(
            r'[̥▽☞➕」ﾟ*のε√Θ☝⑉°‍€✂̷̤¥​ꉂ‹͈™▔★ô⭐⚠è☑•ਊ⌔＜αὼ☔̴ˊິ͒⚡✳❤›♥⏰でé＃￥⬅￼♀#✌☁ꎿ⑤／½＆౨☄¯♪٩ˋ➡❄⭕●『️×☕✏ᗨᵒꇴェ✔✈「✨༝’✠∠③＠❣೭⛄ง②´◎✊♔』̶з⃣ㄒ→ຶ–＄́㊙ȏ５✅｀՞⌯˵π☀♨✍❌ᵕ①▪☺⚜ᐢ￡ੈ⛳＊۶Ⅱ⚽๑⛱꒦$℃〃♓⑥∩④□Дβ੭∧]')
        text_s = ta.sub('', text_s)
        text_s = re.sub('^[,，：:.。！ !]+|[,，：: ]+$', '', text_s)
        text_s = re.sub('哈', '', text_s)
    else:
        text_s = str(text)
        text_s = re_sub(text_s)
    return text_s


def sent_split(doc):
    try:
        doc = str(doc)
    except:
        logger.error('`doc` can not be convert to a str to carry the work below.', exc_info=True)
    sents = pyltp.SentenceSplitter.split(doc)
    return sents


filter_sent_by_kw = lambda sent, kw: sent if re.search(kw, sent) else None


def dedup_sent(doc, filter_func=lambda x: x):
    return list(set([filter_func(sent) for sent in sent_split(doc)]))


dedup_list2str = lambda doc: ''.join(dedup_sent(doc))


def filter_short_text(text, txt_len):
    if isinstance(text, pd.core.series.Series):
        content_len = text.map(lambda x: len(x.strip()))
        return content_len > txt_len
    else:
        logger.error('`text` is not a pandas Series object.', exc_info=True)
        raise ValueError('Error: `text` is not a pandas Series object.')


def text_clean(text_df, _DOC_LENGTH_FFILTER, **columns):
    """
    过滤了非法字符、短文本、特殊字符
    """
    if isinstance(text_df, pd.core.frame.DataFrame):
        text_df[columns['content']] = text_df[columns['content']].map(lambda x: x if isinstance(x, str) else str(x))
        text_df[columns['content']] = text_df[columns['content']].map(lambda x: re.sub(u'\u200b', '', x))
        text_df[columns['content']] = text_df[columns['content']].map(illegal_char_remover)
        text_df[columns['content']] = text_df[columns['content']].map(re_sub)
        text_df[columns['content']] = text_df[columns['content']].map(dedup_list2str)
        text_df = text_df.loc[filter_short_text(text_df[columns['content']], _DOC_LENGTH_FFILTER)]
        return text_df
    else:
        logger.error('`text` is not a pandas Series object.', exc_info=True)
        raise ValueError('Error: `text` is not a pandas Series object.')

