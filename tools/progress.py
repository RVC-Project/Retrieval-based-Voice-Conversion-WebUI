import math

from i18n.i18n import I18nAuto


i18n = I18nAuto()


def should_report(index, total, max_updates=12):
    if total <= 0:
        return False
    if total <= max_updates:
        return True
    interval = max(1, math.ceil(total / max_updates))
    return index == 0 or index + 1 == total or (index + 1) % interval == 0


def batch_status(title, current, total, success, failed, latest="", failures=None):
    if total <= 0:
        state = i18n("等待输入")
    elif current >= total:
        state = i18n("已完成")
    else:
        state = i18n("处理中")
    lines = [
        "【%s】" % title,
        "%s：%s" % (i18n("状态"), state),
        "%s：%s/%s | %s：%s | %s：%s"
        % (
            i18n("进度"),
            current,
            total,
            i18n("成功"),
            success,
            i18n("失败"),
            failed,
        ),
    ]
    if latest:
        lines.append("%s：%s" % (i18n("当前"), latest))
    if failures:
        lines.append("%s：" % i18n("失败记录"))
        lines.extend(failures[-10:])
        if len(failures) > 10:
            lines.append(i18n("……仅显示最近10条失败记录"))
    return "\n".join(lines)
