import React, { useState } from 'react';
import { Card, Button, Form, Input, Select, DatePicker, message, Table, Tag } from 'antd';
import { PlayCircleOutlined, StopOutlined } from '@ant-design/icons';

const { RangePicker } = DatePicker;
const { Option } = Select;

const Collection: React.FC = () => {
  const [loading, setLoading] = useState(false);

  const onFinish = async (values: any) => {
    setLoading(true);
    try {
      console.log('Collect:', values);
      message.success('Сбор данных начат!');
      // TODO: Отправить запрос на backend
    } catch (error) {
      message.error('Ошибка при запуске сбора данных');
    } finally {
      setLoading(false);
    }
  };

  const tasksData = [
    {
      key: '1',
      source: 'ВКонтакте',
      group: 'club123456',
      status: 'active',
      collected: 150,
      date: '2026-01-15 10:00',
    },
    {
      key: '2',
      source: 'Одноклассники',
      group: 'group789',
      status: 'completed',
      collected: 85,
      date: '2026-01-14 15:30',
    },
  ];

  const columns = [
    { title: 'Источник', dataIndex: 'source', key: 'source' },
    { title: 'Группа', dataIndex: 'group', key: 'group' },
    {
      title: 'Статус',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={status === 'active' ? 'green' : 'blue'}>
          {status === 'active' ? 'Активен' : 'Завершен'}
        </Tag>
      ),
    },
    { title: 'Собрано', dataIndex: 'collected', key: 'collected' },
    { title: 'Дата', dataIndex: 'date', key: 'date' },
  ];

  return (
    <div style={{ padding: 24 }}>
      <h1>Сбор данных</h1>

      <Card title="Настройка сбора отзывов" style={{ marginBottom: 24 }}>
        <Form onFinish={onFinish} layout="vertical">
          <Form.Item
            label="Источник"
            name="source"
            rules={[{ required: true, message: 'Выберите источник' }]}
          >
            <Select placeholder="Выберите источник">
              <Option value="vk">ВКонтакте</Option>
              <Option value="ok">Одноклассники</Option>
              <Option value="both">Оба источника</Option>
            </Select>
          </Form.Item>

          <Form.Item
            label="ID или ссылка группы ВКонтакте"
            name="vk_group"
            rules={[{ required: false }]}
          >
            <Input placeholder="Например: club123456 или https://vk.com/club123456" />
          </Form.Item>

          <Form.Item
            label="ID или ссылка группы Одноклассники"
            name="ok_group"
            rules={[{ required: false }]}
          >
            <Input placeholder="Например: group789 или https://ok.ru/group/789" />
          </Form.Item>

          <Form.Item
            label="Период сбора"
            name="date_range"
            rules={[{ required: true, message: 'Выберите период' }]}
          >
            <RangePicker style={{ width: '100%' }} />
          </Form.Item>

          <Form.Item>
            <Button
              type="primary"
              htmlType="submit"
              icon={<PlayCircleOutlined />}
              loading={loading}
              block
            >
              Начать сбор
            </Button>
          </Form.Item>
        </Form>
      </Card>

      <Card title="Активные задачи сбора">
        <Table dataSource={tasksData} columns={columns} pagination={false} />
      </Card>
    </div>
  );
};

export default Collection;