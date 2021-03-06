from datetime import datetime


def main():
    lessons = [(3, '21:30', '23:45'), (5, '18:30', '20:35')]
    date_format = '%Y:%m:%d:%H:%M'
    out_format = "%d/%m/%Y  %H:%M"
    per_hour = 16
    for day, l1, l2 in lessons:
        d1 = datetime.strptime(f"2021:03:{day}:{l1}", date_format)
        import ipdb
        # ipdb.set_trace()
        d2 = datetime.strptime(f"2021:03:{day}:{l2}", date_format)
        print(d1.strftime(out_format))
        print(d2.strftime(out_format))
        hour_diff = (d2 - d1).total_seconds() / (60 * 60)

        num = float('0.' + str(hour_diff).split('.')[1])
        # print(len(str(num)) - 2)
        # num = num * 10**(-len(str(num)) + 2)
        minutes = round(60 * num)
        print(
            f"Hours: {round(hour_diff)}:{minutes if len(str(minutes)) > 1 else '0'+str(minutes)}"
        )
        print(
            f"{round(hour_diff, 2)}hx€{per_hour} = {round(per_hour*hour_diff, 2)}\n"
        )


if __name__ == '__main__':
    main()
